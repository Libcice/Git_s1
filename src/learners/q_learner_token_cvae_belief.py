import copy
import os

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
from torch.optim import RMSprop


class QLearnerTokenCVAEBelief:
    """QMIX learner for token-history + CVAE belief."""

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.unit_type_bits = max(int(getattr(self.args, "unit_dim", 5)) - 5, 0)
        self.enemy_state_feat_dim = getattr(self.args, "enemy_state_feat_dim", 4 + self.unit_type_bits)
        self.enemy_state_dim = getattr(self.args, "enemy_state_dim", self.args.enemy_num * self.enemy_state_feat_dim)
        self.ally_state_feat_dim = 5 + self.unit_type_bits
        self.ally_state_dim = self.args.n_agents * self.ally_state_feat_dim
        env_args = getattr(self.args, "env_args", {})
        self.move_dim = 4
        if env_args.get("obs_pathing_grid", False):
            self.move_dim += 8
        if env_args.get("obs_terrain_height", False):
            self.move_dim += 9
        health_bits = 1 + 1 if env_args.get("obs_all_health", True) else 0
        self.enemy_obs_feat_dim = 4 + self.unit_type_bits + health_bits

        self.disable_belief = getattr(self.args, "disable_belief", False)
        self.use_belief_for_q = getattr(self.args, "use_belief_for_q", True) and not self.disable_belief
        self.belief_prior_type = getattr(self.args, "belief_prior_type", "conditional")
        self.belief_loss_coef = getattr(self.args, "belief_loss_coef", 0.05)
        self.belief_kl_coef = getattr(self.args, "belief_kl_coef", 0.1)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -3.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 300000)
        self.belief_pretrain_t = getattr(self.args, "belief_pretrain_t", 50000)
        self.belief_pretrain_only = getattr(self.args, "belief_pretrain_only", False)
        if self.disable_belief:
            self.belief_loss_coef = 0.0
            self.belief_kl_coef = 0.0
            self.belief_pretrain_only = False
        self.belief_enabled = (self.belief_loss_coef > 0.0) or self.use_belief_for_q
        self.posterior_enabled = self.belief_loss_coef > 0.0
        self.q_learning_started = False

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _gaussian_nll(self, target, mu, logvar):
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        inv_var = th.exp(-logvar)
        return 0.5 * (((target - mu) ** 2) * inv_var + logvar).sum(dim=-1)

    def _kl_divergence(self, post_mu, post_logvar, prior_mu, prior_logvar):
        post_logvar = post_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        prior_logvar = prior_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)

        if self.belief_prior_type == "standard_normal":
            return 0.5 * (th.exp(post_logvar) + post_mu.pow(2) - 1.0 - post_logvar).sum(dim=-1)

        diff = post_mu - prior_mu
        return 0.5 * (
            prior_logvar - post_logvar + (th.exp(post_logvar) + diff.pow(2)) * th.exp(-prior_logvar) - 1.0
        ).sum(dim=-1)

    def _mean_or_zero(self, tensor):
        if tensor is None:
            return 0.0
        return tensor.mean().item()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        belief_pretraining_active = (self.belief_loss_coef > 0.0) and (t_env < self.belief_pretrain_t)
        q_learning_active = (not self.belief_pretrain_only) or (t_env >= self.belief_pretrain_t)
        if q_learning_active and not self.q_learning_started:
            self._update_targets()
            self.q_learning_started = True
        if hasattr(self.mac, "set_belief_q_alpha"):
            self.mac.set_belief_q_alpha(1.0)
        if hasattr(self.target_mac, "set_belief_q_alpha"):
            self.target_mac.set_belief_q_alpha(1.0)

        mac_out = []
        belief_recon_sum = rewards.new_tensor(0.0)
        belief_recon_denom = rewards.new_tensor(0.0)
        belief_prior_nll_sum = rewards.new_tensor(0.0)
        belief_prior_nll_denom = rewards.new_tensor(0.0)
        belief_kl_sum = rewards.new_tensor(0.0)
        belief_kl_denom = rewards.new_tensor(0.0)
        prior_state_mu_sum = rewards.new_tensor(0.0)
        prior_state_mu_count = rewards.new_tensor(0.0)
        prior_state_logvar_sum = rewards.new_tensor(0.0)
        prior_state_logvar_count = rewards.new_tensor(0.0)
        post_state_mu_sum = rewards.new_tensor(0.0)
        post_state_mu_count = rewards.new_tensor(0.0)
        post_state_logvar_sum = rewards.new_tensor(0.0)
        post_state_logvar_count = rewards.new_tensor(0.0)
        prior_z_mu_sum = rewards.new_tensor(0.0)
        prior_z_mu_count = rewards.new_tensor(0.0)
        prior_z_logvar_sum = rewards.new_tensor(0.0)
        prior_z_logvar_count = rewards.new_tensor(0.0)
        post_z_mu_sum = rewards.new_tensor(0.0)
        post_z_mu_count = rewards.new_tensor(0.0)
        post_z_logvar_sum = rewards.new_tensor(0.0)
        post_z_logvar_count = rewards.new_tensor(0.0)
        prior_conf_sum = rewards.new_tensor(0.0)
        prior_conf_count = rewards.new_tensor(0.0)

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim
        hidden_enemy_state_active = self.posterior_enabled or belief_pretraining_active

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            hidden_enemy_state_t = None
            if hidden_enemy_state_active:
                enemy_state_t = batch["state"][:, t, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
                enemy_state_t = enemy_state_t.view(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
                hidden_enemy_state_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)
                hidden_enemy_state_t = hidden_enemy_state_t.reshape(
                    batch.batch_size * self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )

            agent_outs = self.mac.forward(batch, t=t, hidden_enemy_state=hidden_enemy_state_t)
            mac_out.append(agent_outs)

            if not self.belief_enabled:
                continue

            prior_state_mu_t, prior_state_logvar_t = self.mac.get_prior_belief_stats()
            prior_z_mu_t, prior_z_logvar_t = self.mac.get_prior_latent_stats()
            prior_conf_t = self.mac.get_prior_belief_confidence()

            prior_state_mu_sum = prior_state_mu_sum + prior_state_mu_t.mean()
            prior_state_mu_count = prior_state_mu_count + prior_state_mu_t.new_tensor(1.0)
            prior_state_logvar_sum = prior_state_logvar_sum + prior_state_logvar_t.mean()
            prior_state_logvar_count = prior_state_logvar_count + prior_state_logvar_t.new_tensor(1.0)
            prior_z_mu_sum = prior_z_mu_sum + prior_z_mu_t.mean()
            prior_z_mu_count = prior_z_mu_count + prior_z_mu_t.new_tensor(1.0)
            prior_z_logvar_sum = prior_z_logvar_sum + prior_z_logvar_t.mean()
            prior_z_logvar_count = prior_z_logvar_count + prior_z_logvar_t.new_tensor(1.0)
            prior_conf_sum = prior_conf_sum + prior_conf_t.mean()
            prior_conf_count = prior_conf_count + prior_conf_t.new_tensor(1.0)

            if hidden_enemy_state_active:
                posterior_state_mu_t, posterior_state_logvar_t = self.mac.get_posterior_belief_stats()
                posterior_z_mu_t, posterior_z_logvar_t = self.mac.get_posterior_latent_stats()
                post_state_mu_sum = post_state_mu_sum + posterior_state_mu_t.mean()
                post_state_mu_count = post_state_mu_count + posterior_state_mu_t.new_tensor(1.0)
                post_state_logvar_sum = post_state_logvar_sum + posterior_state_logvar_t.mean()
                post_state_logvar_count = post_state_logvar_count + posterior_state_logvar_t.new_tensor(1.0)
                post_z_mu_sum = post_z_mu_sum + posterior_z_mu_t.mean()
                post_z_mu_count = post_z_mu_count + posterior_z_mu_t.new_tensor(1.0)
                post_z_logvar_sum = post_z_logvar_sum + posterior_z_logvar_t.mean()
                post_z_logvar_count = post_z_logvar_count + posterior_z_logvar_t.new_tensor(1.0)

                enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
                enemy_obs_t = enemy_obs_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_obs_feat_dim,
                )
                enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
                alive_mask_t = (hidden_enemy_state_t.view(batch.batch_size, self.args.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)[..., 0] > 0).float()
                time_mask_t = batch["filled"][:, t].float().unsqueeze(1).expand(-1, self.args.n_agents, -1)
                hidden_mask_t = (1.0 - enemy_visible_t) * alive_mask_t * time_mask_t
                hidden_present_t = (hidden_mask_t.sum(dim=-1) > 0).float()

                posterior_nll_t = self._gaussian_nll(hidden_enemy_state_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                ), posterior_state_mu_t, posterior_state_logvar_t)
                prior_nll_t = self._gaussian_nll(hidden_enemy_state_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                ), prior_state_mu_t, prior_state_logvar_t)
                belief_recon_sum = belief_recon_sum + (posterior_nll_t * hidden_mask_t).sum()
                belief_recon_denom = belief_recon_denom + hidden_mask_t.sum()
                belief_prior_nll_sum = belief_prior_nll_sum + (prior_nll_t * hidden_mask_t).sum()
                belief_prior_nll_denom = belief_prior_nll_denom + hidden_mask_t.sum()

                kl_t = self._kl_divergence(posterior_z_mu_t, posterior_z_logvar_t, prior_z_mu_t, prior_z_logvar_t)
                belief_kl_sum = belief_kl_sum + (kl_t * hidden_present_t).sum()
                belief_kl_denom = belief_kl_denom + hidden_present_t.sum()

        mac_out = th.stack(mac_out, dim=1)

        if q_learning_active:
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            if self.mixer is not None:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                with th.no_grad():
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            n_step = getattr(self.args, "n_step", 1)
            if n_step == 1:
                targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
            else:
                n_rewards = th.zeros_like(rewards)
                gamma_tensor = th.tensor(
                    [self.args.gamma ** i for i in range(n_step)],
                    dtype=th.float,
                    device=rewards.device,
                )
                steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(n_step).long()
                for i in range(batch.max_seq_length - 1):
                    horizon = min(n_step, batch.max_seq_length - 1 - i)
                    n_rewards[:, i, 0] = ((rewards * mask)[:, i:i + horizon, 0] * gamma_tensor[:horizon]).sum(dim=1)
                indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
                n_targets_terminated = th.gather(target_max_qvals * (1 - terminated), dim=1, index=steps + indices - 1)
                targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

            td_error = chosen_action_qvals - targets.detach()
            q_mask = mask.expand_as(td_error)
            masked_td_error = td_error * q_mask
            q_loss = (masked_td_error ** 2).sum() / q_mask.sum()
        else:
            chosen_action_qvals = None
            targets = None
            td_error = None
            q_mask = None
            masked_td_error = None
            q_loss = rewards.new_tensor(0.0)

        if belief_pretraining_active:
            belief_weight = 1.0
        else:
            warmup_progress = min(
                1.0,
                float(max(1, t_env - self.belief_pretrain_t + 1)) / float(max(1, self.belief_warmup_t)),
            )
            belief_weight = self.belief_loss_coef * warmup_progress if self.belief_loss_coef > 0.0 else 0.0

        belief_recon_loss = belief_recon_sum / belief_recon_denom.clamp(min=1.0)
        belief_prior_nll = belief_prior_nll_sum / belief_prior_nll_denom.clamp(min=1.0)
        belief_kl_loss = belief_kl_sum / belief_kl_denom.clamp(min=1.0)
        belief_loss = belief_recon_loss + self.belief_kl_coef * belief_kl_loss
        loss = q_loss + belief_weight * belief_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if q_learning_active and (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_recon_loss", belief_recon_loss.item(), t_env)
            self.logger.log_stat("belief_prior_nll", belief_prior_nll.item(), t_env)
            self.logger.log_stat("belief_kl_loss", belief_kl_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_kl_coef", self.belief_kl_coef, t_env)
            self.logger.log_stat("belief_pretraining_active", float(belief_pretraining_active), t_env)
            self.logger.log_stat("belief_pretrain_only", float(self.belief_pretrain_only), t_env)
            self.logger.log_stat("belief_prior_state_mu_mean", prior_state_mu_sum.item() / prior_state_mu_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_state_logvar_mean", prior_state_logvar_sum.item() / prior_state_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_z_mu_mean", prior_z_mu_sum.item() / prior_z_mu_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_z_logvar_mean", prior_z_logvar_sum.item() / prior_z_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_confidence_mean", prior_conf_sum.item() / prior_conf_count.clamp(min=1.0).item(), t_env)
            if hidden_enemy_state_active:
                self.logger.log_stat("belief_post_state_mu_mean", post_state_mu_sum.item() / post_state_mu_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_state_logvar_mean", post_state_logvar_sum.item() / post_state_logvar_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_z_mu_mean", post_z_mu_sum.item() / post_z_mu_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_z_logvar_mean", post_z_logvar_sum.item() / post_z_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if q_learning_active:
                mask_elems = q_mask.sum().item()
                self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals * q_mask).sum().item() / (mask_elems * self.args.n_agents),
                    t_env,
                )
                self.logger.log_stat(
                    "target_mean",
                    (targets * q_mask).sum().item() / (mask_elems * self.args.n_agents),
                    t_env,
                )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        opt_state = th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        try:
            self.optimiser.load_state_dict(opt_state)
        except ValueError:
            pass
