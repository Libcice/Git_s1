import copy
import math

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerTokenFutureBelief:
    """QMIX learner for action-conditioned value-aware future belief."""

    def __init__(self, mac, scheme, logger, args):
        del scheme
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
        self.belief_loss_coef = getattr(self.args, "belief_loss_coef", 0.01)
        self.belief_current_coef = getattr(self.args, "belief_current_coef", 1.0)
        self.belief_future_coef = getattr(self.args, "belief_future_coef", 1.0)
        self.belief_posterior_coef = getattr(self.args, "belief_posterior_coef", 0.25)
        self.belief_kl_coef = getattr(self.args, "belief_kl_coef", 0.1)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -3.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_nll_clip = getattr(self.args, "belief_nll_clip", 2.0)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 300000)
        self.belief_kl_warmup_t = getattr(self.args, "belief_kl_warmup_t", self.belief_warmup_t)
        self.belief_pretrain_t = getattr(self.args, "belief_pretrain_t", 50000)
        self.belief_pretrain_only = getattr(self.args, "belief_pretrain_only", False)
        self.belief_future_q_warmup_t = getattr(self.args, "belief_future_q_warmup_t", 600000)
        self.future_repr_mask = getattr(self.args, "future_repr_mask", "next_hidden")
        self.future_delta_value_coef = getattr(self.args, "future_delta_value_coef", 0.0)
        if self.disable_belief:
            self.belief_loss_coef = 0.0
            self.belief_current_coef = 0.0
            self.belief_future_coef = 0.0
            self.belief_kl_coef = 0.0
            self.future_delta_value_coef = 0.0
            self.belief_pretrain_only = False
        self.belief_enabled = (self.belief_loss_coef > 0.0) or self.use_belief_for_q
        self.posterior_enabled = self.belief_loss_coef > 0.0
        self.q_learning_started = False
        self._replay_reset_done = not self.belief_pretrain_only

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _smooth_ramp(self, progress):
        progress = max(0.0, min(1.0, float(progress)))
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def _stabilized_gaussian_nll(self, target, mu, logvar):
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        inv_var = th.exp(-logvar)
        per_dim_nll = 0.5 * (((target - mu) ** 2) * inv_var + logvar + math.log(2.0 * math.pi))
        raw_nll = per_dim_nll.mean(dim=-1)
        nll_floor = 0.5 * (self.belief_logvar_min + math.log(2.0 * math.pi))
        stable_nll = (raw_nll - nll_floor).clamp(min=0.0, max=self.belief_nll_clip)
        return raw_nll, stable_nll

    def _kl_divergence(self, post_mu, post_logvar, prior_mu, prior_logvar):
        post_logvar = post_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        prior_logvar = prior_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)

        if self.belief_prior_type == "standard_normal":
            kl = 0.5 * (th.exp(post_logvar) + post_mu.pow(2) - 1.0 - post_logvar).sum(dim=-1)
        else:
            diff = post_mu - prior_mu
            kl = 0.5 * (
                prior_logvar - post_logvar + (th.exp(post_logvar) + diff.pow(2)) * th.exp(-prior_logvar) - 1.0
            ).sum(dim=-1)

        latent_dim = float(max(1, getattr(self.args, "belief_latent_dim", 1)))
        return kl / latent_dim

    def _gather_action_values(self, values, actions_t):
        return th.gather(values, dim=2, index=actions_t.long()).squeeze(2)

    def _gather_action_repr(self, values, actions_t):
        index = actions_t.long().unsqueeze(-1).unsqueeze(-1)
        index = index.expand(-1, -1, -1, values.size(3), values.size(4))
        return th.gather(values, dim=2, index=index).squeeze(2)

    def should_clear_replay_buffer(self, t_env: int):
        if not self.belief_pretrain_only:
            return False
        if self._replay_reset_done:
            return False
        if t_env < self.belief_pretrain_t:
            return False
        self._replay_reset_done = True
        return True

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        q_learning_active = (not self.belief_pretrain_only) or (t_env >= self.belief_pretrain_t)
        if q_learning_active and not self.q_learning_started:
            self._update_targets()
            self.q_learning_started = True

        if self.use_belief_for_q:
            future_q_alpha = self._smooth_ramp(float(max(0, t_env)) / float(max(1, self.belief_future_q_warmup_t)))
        else:
            future_q_alpha = 0.0
        if hasattr(self.mac, "set_future_q_alpha"):
            self.mac.set_future_q_alpha(future_q_alpha)
        if hasattr(self.target_mac, "set_future_q_alpha"):
            self.target_mac.set_future_q_alpha(future_q_alpha)

        mac_out = []
        current_prior_sum = rewards.new_tensor(0.0)
        current_prior_denom = rewards.new_tensor(0.0)
        current_post_sum = rewards.new_tensor(0.0)
        current_post_denom = rewards.new_tensor(0.0)
        current_kl_sum = rewards.new_tensor(0.0)
        current_kl_denom = rewards.new_tensor(0.0)
        future_repr_sum = rewards.new_tensor(0.0)
        future_repr_denom = rewards.new_tensor(0.0)
        hidden_frac_sum = rewards.new_tensor(0.0)
        hidden_frac_count = rewards.new_tensor(0.0)
        next_hidden_frac_sum = rewards.new_tensor(0.0)
        next_hidden_frac_count = rewards.new_tensor(0.0)
        current_conf_sum = rewards.new_tensor(0.0)
        current_conf_count = rewards.new_tensor(0.0)
        current_base_gate_sum = rewards.new_tensor(0.0)
        current_base_gate_count = rewards.new_tensor(0.0)
        future_gate_sum = rewards.new_tensor(0.0)
        future_gate_count = rewards.new_tensor(0.0)
        future_selected_gate_sum = rewards.new_tensor(0.0)
        future_selected_gate_count = rewards.new_tensor(0.0)
        future_delta_abs_sum = rewards.new_tensor(0.0)
        future_delta_abs_count = rewards.new_tensor(0.0)
        future_weighted_abs_sum = rewards.new_tensor(0.0)
        future_base_abs_sum = rewards.new_tensor(0.0)

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
            enemy_obs_t = enemy_obs_t.view(
                batch.batch_size,
                self.args.n_agents,
                self.args.enemy_num,
                self.enemy_obs_feat_dim,
            )
            enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
            enemy_state_t = batch["state"][:, t, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
            enemy_state_t = enemy_state_t.view(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
            enemy_state_agents_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)

            alive_mask_t = (enemy_state_agents_t[..., 0] > 0).float()
            time_mask_t = batch["filled"][:, t].float().unsqueeze(1).expand(-1, self.args.n_agents, -1)
            active_agent_mask_t = time_mask_t.squeeze(-1)
            alive_present_t = (alive_mask_t.sum(dim=-1) > 0).float()
            active_visible_mask_t = active_agent_mask_t * alive_present_t
            alive_total_t = alive_mask_t.sum(dim=-1).clamp(min=1.0)
            current_hidden_mask_t = (1.0 - enemy_visible_t) * alive_mask_t * time_mask_t
            hidden_frac_t = current_hidden_mask_t.sum(dim=-1) / alive_total_t
            hidden_frac_sum = hidden_frac_sum + (hidden_frac_t * active_visible_mask_t).sum()
            hidden_frac_count = hidden_frac_count + active_visible_mask_t.sum()

            hidden_enemy_state_t = None
            if self.posterior_enabled:
                hidden_enemy_state_t = enemy_state_agents_t.reshape(
                    batch.batch_size * self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )

            agent_outs = self.mac.forward(
                batch,
                t=t,
                hidden_enemy_state=hidden_enemy_state_t,
                selected_actions=actions[:, t] if t < batch.max_seq_length - 1 else None,
            )
            mac_out.append(agent_outs)

            if not self.belief_enabled:
                continue

            current_prior_conf_t = self.mac.get_current_prior_confidence()
            current_base_gate_t = self.mac.get_current_base_gate().squeeze(-1)
            future_gate_t = self.mac.get_future_gate()
            future_delta_q_t = self.mac.get_future_delta_q()
            future_weighted_delta_q_t = self.mac.get_future_weighted_delta_q()
            base_q_t = self.mac.get_base_q()

            current_conf_sum = current_conf_sum + current_prior_conf_t.mean()
            current_conf_count = current_conf_count + current_prior_conf_t.new_tensor(1.0)
            current_base_gate_sum = current_base_gate_sum + (current_base_gate_t * current_hidden_mask_t).sum()
            current_base_gate_count = current_base_gate_count + current_hidden_mask_t.sum()

            active_action_mask_t = active_agent_mask_t.unsqueeze(-1).expand_as(future_gate_t)
            future_gate_sum = future_gate_sum + (future_gate_t * active_action_mask_t).sum()
            future_gate_count = future_gate_count + active_action_mask_t.sum()
            future_delta_abs_sum = future_delta_abs_sum + (future_delta_q_t.abs() * active_action_mask_t).sum()
            future_delta_abs_count = future_delta_abs_count + active_action_mask_t.sum()
            future_weighted_abs_sum = future_weighted_abs_sum + (future_weighted_delta_q_t.abs() * active_action_mask_t).sum()
            future_base_abs_sum = future_base_abs_sum + (base_q_t.abs() * active_action_mask_t).sum()

            if t < batch.max_seq_length - 1:
                selected_gate_t = self._gather_action_values(future_gate_t, actions[:, t])
                future_selected_gate_sum = future_selected_gate_sum + (selected_gate_t * active_agent_mask_t).sum()
                future_selected_gate_count = future_selected_gate_count + active_agent_mask_t.sum()

            if self.posterior_enabled:
                current_prior_mu_t, current_prior_logvar_t = self.mac.get_current_prior_belief_stats()
                current_post_mu_t, current_post_logvar_t = self.mac.get_current_posterior_belief_stats()
                current_prior_z_mu_t, current_prior_z_logvar_t = self.mac.get_current_prior_latent_stats()
                current_post_z_mu_t, current_post_z_logvar_t = self.mac.get_current_posterior_latent_stats()

                _, current_prior_nll_t = self._stabilized_gaussian_nll(
                    enemy_state_agents_t,
                    current_prior_mu_t,
                    current_prior_logvar_t,
                )
                _, current_post_nll_t = self._stabilized_gaussian_nll(
                    enemy_state_agents_t,
                    current_post_mu_t,
                    current_post_logvar_t,
                )
                current_prior_sum = current_prior_sum + (current_prior_nll_t * current_hidden_mask_t).sum()
                current_prior_denom = current_prior_denom + current_hidden_mask_t.sum()
                current_post_sum = current_post_sum + (current_post_nll_t * current_hidden_mask_t).sum()
                current_post_denom = current_post_denom + current_hidden_mask_t.sum()

                current_hidden_present_t = (current_hidden_mask_t.sum(dim=-1) > 0).float()
                current_kl_t = self._kl_divergence(
                    current_post_z_mu_t,
                    current_post_z_logvar_t,
                    current_prior_z_mu_t,
                    current_prior_z_logvar_t,
                )
                current_kl_sum = current_kl_sum + (current_kl_t * current_hidden_present_t).sum()
                current_kl_denom = current_kl_denom + current_hidden_present_t.sum()

            if (not self.posterior_enabled) or t + 1 >= batch.max_seq_length:
                continue

            next_enemy_state_t = batch["state"][:, t + 1, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
            next_enemy_state_t = next_enemy_state_t.view(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
            next_enemy_state_agents_t = next_enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)
            next_time_mask_t = batch["filled"][:, t + 1].float().unsqueeze(1).expand(-1, self.args.n_agents, -1)
            next_enemy_obs_t = batch["obs"][:, t + 1, :, enemy_obs_start:enemy_obs_end]
            next_enemy_obs_t = next_enemy_obs_t.view(
                batch.batch_size,
                self.args.n_agents,
                self.args.enemy_num,
                self.enemy_obs_feat_dim,
            )
            next_enemy_visible_t = (next_enemy_obs_t.abs().sum(dim=-1) > 0).float()
            next_alive_mask_t = (next_enemy_state_agents_t[..., 0] > 0).float()
            transition_time_mask_t = time_mask_t * next_time_mask_t * (1 - batch["terminated"][:, t].float()).unsqueeze(1)
            next_hidden_mask_t = (1.0 - next_enemy_visible_t) * next_alive_mask_t * transition_time_mask_t
            next_alive_total_t = next_alive_mask_t.sum(dim=-1).clamp(min=1.0)
            next_hidden_frac_t = next_hidden_mask_t.sum(dim=-1) / next_alive_total_t
            next_alive_present_t = (next_alive_mask_t.sum(dim=-1) > 0).float()
            active_next_mask_t = active_agent_mask_t * next_alive_present_t
            next_hidden_frac_sum = next_hidden_frac_sum + (next_hidden_frac_t * active_next_mask_t).sum()
            next_hidden_frac_count = next_hidden_frac_count + active_next_mask_t.sum()

            hidden_union_mask_t = (
                (((1.0 - enemy_visible_t) * alive_mask_t) + ((1.0 - next_enemy_visible_t) * next_alive_mask_t)) > 0
            ).float()
            hidden_union_mask_t = hidden_union_mask_t * ((alive_mask_t + next_alive_mask_t) > 0).float() * transition_time_mask_t
            if self.future_repr_mask == "hidden_union":
                future_repr_mask_t = hidden_union_mask_t
            else:
                future_repr_mask_t = next_hidden_mask_t

            selected_future_repr_t = self.mac.get_selected_future_repr()
            if selected_future_repr_t is None:
                continue
            with th.no_grad():
                future_target_repr_t = self.target_mac.agent.future_target_proj(
                    F.relu(
                        self.target_mac.agent.current_belief_feat_proj(
                            next_enemy_state_agents_t.reshape(-1, self.enemy_state_feat_dim)
                        )
                    )
                ).view(batch.batch_size, self.args.n_agents, self.args.enemy_num, -1)

            future_repr_loss_t = F.smooth_l1_loss(
                selected_future_repr_t,
                future_target_repr_t,
                reduction="none",
            ).mean(dim=-1)
            future_repr_sum = future_repr_sum + (future_repr_loss_t * future_repr_mask_t).sum()
            future_repr_denom = future_repr_denom + future_repr_mask_t.sum()

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
            q_loss = (masked_td_error ** 2).sum() / q_mask.sum().clamp(min=1.0)
        else:
            chosen_action_qvals = None
            targets = None
            q_mask = None
            masked_td_error = None
            q_loss = rewards.new_tensor(0.0)

        kl_progress = min(1.0, float(max(1, t_env)) / float(max(1, self.belief_kl_warmup_t)))
        belief_kl_weight = self.belief_kl_coef * self._smooth_ramp(kl_progress)
        if self.belief_pretrain_only:
            if t_env < self.belief_pretrain_t:
                belief_weight = self._smooth_ramp(float(max(0, t_env)) / float(max(1, self.belief_pretrain_t)))
            else:
                post_progress = min(
                    1.0,
                    float(max(0, t_env - self.belief_pretrain_t)) / float(max(1, self.belief_warmup_t)),
                )
                belief_weight = self.belief_loss_coef + (1.0 - self.belief_loss_coef) * (1.0 - self._smooth_ramp(post_progress))
        else:
            warmup_progress = min(1.0, float(max(0, t_env)) / float(max(1, self.belief_warmup_t)))
            belief_weight = self.belief_loss_coef * self._smooth_ramp(warmup_progress) if self.belief_loss_coef > 0.0 else 0.0

        current_prior_loss = current_prior_sum / current_prior_denom.clamp(min=1.0)
        current_post_loss = current_post_sum / current_post_denom.clamp(min=1.0)
        current_kl_loss = current_kl_sum / current_kl_denom.clamp(min=1.0)
        future_selected_repr_loss = future_repr_sum / future_repr_denom.clamp(min=1.0)
        future_delta_value_loss = rewards.new_tensor(0.0)

        current_belief_loss = current_prior_loss + self.belief_posterior_coef * current_post_loss + belief_kl_weight * current_kl_loss
        future_belief_loss = future_selected_repr_loss + self.future_delta_value_coef * future_delta_value_loss
        belief_loss = self.belief_current_coef * current_belief_loss + self.belief_future_coef * future_belief_loss
        belief_weighted_loss = belief_weight * belief_loss
        loss = q_loss + belief_weighted_loss

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
            self.logger.log_stat("belief_weighted_loss", belief_weighted_loss.item(), t_env)
            self.logger.log_stat(
                "belief_weighted_q_ratio",
                abs(belief_weighted_loss.item()) / (q_loss.item() + 1e-6) if q_learning_active else 0.0,
                t_env,
            )
            self.logger.log_stat("belief_current_loss", current_belief_loss.item(), t_env)
            self.logger.log_stat("belief_curr_prior_loss", current_prior_loss.item(), t_env)
            self.logger.log_stat("belief_curr_post_loss", current_post_loss.item(), t_env)
            self.logger.log_stat("belief_curr_kl_loss", current_kl_loss.item(), t_env)
            self.logger.log_stat("future_selected_repr_loss", future_selected_repr_loss.item(), t_env)
            self.logger.log_stat("future_delta_value_loss", future_delta_value_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_effective_kl_weight", belief_weight * belief_kl_weight, t_env)
            self.logger.log_stat("future_alpha", future_q_alpha, t_env)
            self.logger.log_stat("future_q_alpha", future_q_alpha, t_env)
            self.logger.log_stat(
                "belief_alive_hidden_frac_mean",
                hidden_frac_sum.item() / hidden_frac_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_next_hidden_frac_mean",
                next_hidden_frac_sum.item() / next_hidden_frac_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_current_prior_confidence_mean",
                current_conf_sum.item() / current_conf_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_current_base_gate_hidden_mean",
                current_base_gate_sum.item() / current_base_gate_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "future_gate_mean",
                future_gate_sum.item() / future_gate_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "future_selected_gate_mean",
                future_selected_gate_sum.item() / future_selected_gate_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "future_delta_abs_mean",
                future_delta_abs_sum.item() / future_delta_abs_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "future_weighted_ratio",
                future_weighted_abs_sum.item() / (future_base_abs_sum.item() + 1e-6),
                t_env,
            )
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if q_learning_active:
                mask_elems = q_mask.sum().item()
                self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / max(mask_elems, 1.0), t_env)
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals * q_mask).sum().item() / (max(mask_elems, 1.0) * self.args.n_agents),
                    t_env,
                )
                self.logger.log_stat(
                    "target_mean",
                    (targets * q_mask).sum().item() / (max(mask_elems, 1.0) * self.args.n_agents),
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
