import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
from torch.optim import RMSprop


class QLearnerHistorySlotBelief:
    """QMIX learner with persistent-slot belief supervision."""

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

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)

        self.unit_type_bits = max(int(getattr(self.args, "unit_dim", 5)) - 5, 0)
        self.enemy_state_feat_dim = getattr(self.args, "enemy_state_feat_dim", 4 + self.unit_type_bits)
        self.enemy_state_dim = getattr(self.args, "enemy_state_dim", self.args.enemy_num * self.enemy_state_feat_dim)
        self.ally_state_dim = self.args.n_agents * (5 + self.unit_type_bits)
        env_args = getattr(self.args, "env_args", {})
        self.move_dim = 4
        if env_args.get("obs_pathing_grid", False):
            self.move_dim += 8
        if env_args.get("obs_terrain_height", False):
            self.move_dim += 9
        health_bits = 1 + 1 if env_args.get("obs_all_health", True) else 0
        self.enemy_obs_feat_dim = 4 + self.unit_type_bits + health_bits

        self.belief_loss_coef = getattr(self.args, "belief_loss_coef", 0.001)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -2.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_nll_clip = getattr(self.args, "belief_nll_clip", 2.0)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 500000)
        self.belief_q_warmup_t = getattr(self.args, "belief_q_warmup_t", 300000)
        self.belief_reappear_coef = getattr(self.args, "belief_reappear_coef", 0.5)
        self.belief_visible_coef = getattr(self.args, "belief_visible_coef", 0.1)
        self.disable_belief_loss = self.belief_loss_coef <= 0.0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _belief_nll(self, state_target, belief_mu, belief_logvar, belief_u):
        diff = state_target - belief_mu
        d_inv = th.exp(-belief_logvar)
        du = d_inv.unsqueeze(-1) * belief_u
        a = th.einsum("...dr,...ds->...rs", belief_u, du)
        rank = a.shape[-1]
        eye = th.eye(rank, device=a.device, dtype=a.dtype).view(*([1] * (a.dim() - 2)), rank, rank)
        a = a + eye

        base_quad = (diff * d_inv * diff).sum(dim=-1)
        u_t_dinv_x = th.einsum("...dr,...d->...r", belief_u, d_inv * diff)
        a_inv_u = th.linalg.solve(a, u_t_dinv_x.unsqueeze(-1)).squeeze(-1)
        corr_quad = (u_t_dinv_x * a_inv_u).sum(dim=-1)
        quad = base_quad - corr_quad

        sign, logdet_a = th.linalg.slogdet(a)
        if not th.all(sign > 0):
            raise RuntimeError("Non-positive definite covariance factor encountered in belief NLL")
        logdet = belief_logvar.sum(dim=-1) + logdet_a
        raw_nll = 0.5 * (quad + logdet)
        nll = raw_nll / float(self.enemy_state_feat_dim)
        nll = nll.clamp(min=0.0, max=self.belief_nll_clip)
        return raw_nll, nll

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.disable_belief_loss:
            belief_q_alpha = 0.0
        else:
            belief_q_alpha = min(1.0, float(t_env) / float(max(1, self.belief_q_warmup_t)))
        if hasattr(self.mac, "set_belief_q_alpha"):
            self.mac.set_belief_q_alpha(belief_q_alpha)
        if hasattr(self.target_mac, "set_belief_q_alpha"):
            self.target_mac.set_belief_q_alpha(belief_q_alpha)

        mac_out = []
        hidden_loss_sum = rewards.new_tensor(0.0)
        hidden_denom = rewards.new_tensor(0.0)
        reappear_loss_sum = rewards.new_tensor(0.0)
        reappear_denom = rewards.new_tensor(0.0)
        visible_loss_sum = rewards.new_tensor(0.0)
        visible_denom = rewards.new_tensor(0.0)
        raw_nll_sum = rewards.new_tensor(0.0)
        raw_nll_count = rewards.new_tensor(0.0)
        belief_logvar_sum = rewards.new_tensor(0.0)
        belief_logvar_count = rewards.new_tensor(0.0)
        belief_mask_sum = rewards.new_tensor(0.0)
        belief_mask_count = rewards.new_tensor(0.0)

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim
        prev_enemy_visible = None

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

            if self.disable_belief_loss:
                continue

            belief_mu_t, belief_logvar_t, belief_u_t = self.mac.get_belief_stats()
            prior_mu_t, prior_logvar_t, prior_u_t = self.mac.get_prior_belief_stats()
            belief_logvar_t = belief_logvar_t.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
            prior_logvar_t = prior_logvar_t.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)

            enemy_state_t = batch["state"][:, t, self.ally_state_dim : self.ally_state_dim + self.enemy_state_dim]
            enemy_state_t = enemy_state_t.view(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
            state_target_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)

            enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
            enemy_obs_t = enemy_obs_t.view(batch.batch_size, self.args.n_agents, self.args.enemy_num, self.enemy_obs_feat_dim)
            enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
            unseen_mask_t = 1.0 - enemy_visible_t
            alive_mask_t = (state_target_t[..., 0] > 0).float()

            if t < batch.max_seq_length - 1:
                time_agent_mask_t = mask[:, t].unsqueeze(1).expand(-1, self.args.n_agents, -1)
                hidden_mask_t = unseen_mask_t * alive_mask_t * time_agent_mask_t
                raw_hidden_t, hidden_nll_t = self._belief_nll(
                    state_target_t,
                    belief_mu_t,
                    belief_logvar_t,
                    belief_u_t,
                )
                hidden_loss_sum = hidden_loss_sum + (hidden_nll_t * hidden_mask_t).sum()
                hidden_denom = hidden_denom + hidden_mask_t.sum()
                raw_nll_sum = raw_nll_sum + raw_hidden_t.sum()
                raw_nll_count = raw_nll_count + raw_hidden_t.new_tensor(float(raw_hidden_t.numel()))
                belief_logvar_sum = belief_logvar_sum + belief_logvar_t.sum()
                belief_logvar_count = belief_logvar_count + belief_logvar_t.new_tensor(float(belief_logvar_t.numel()))
                belief_mask_sum = belief_mask_sum + hidden_mask_t.sum()
                belief_mask_count = belief_mask_count + hidden_mask_t.new_tensor(float(hidden_mask_t.numel()))

                visible_mask_t = enemy_visible_t * alive_mask_t * time_agent_mask_t
                _, visible_nll_t = self._belief_nll(
                    state_target_t,
                    belief_mu_t,
                    belief_logvar_t,
                    belief_u_t,
                )
                visible_loss_sum = visible_loss_sum + (visible_nll_t * visible_mask_t).sum()
                visible_denom = visible_denom + visible_mask_t.sum()

            if prev_enemy_visible is not None:
                reappear_mask_t = (1.0 - prev_enemy_visible) * enemy_visible_t * alive_mask_t
                reappear_time_mask_t = batch["filled"][:, t].float()
                reappear_time_mask_t = reappear_time_mask_t * (1 - batch["terminated"][:, t - 1].float())
                reappear_time_mask_t = reappear_time_mask_t.unsqueeze(1).expand(-1, self.args.n_agents, -1)
                reappear_mask_t = reappear_mask_t * reappear_time_mask_t
                _, reappear_nll_t = self._belief_nll(
                    state_target_t,
                    prior_mu_t,
                    prior_logvar_t,
                    prior_u_t,
                )
                reappear_loss_sum = reappear_loss_sum + (reappear_nll_t * reappear_mask_t).sum()
                reappear_denom = reappear_denom + reappear_mask_t.sum()

            prev_enemy_visible = enemy_visible_t

        mac_out = th.stack(mac_out, dim=1)
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
                n_rewards[:, i, 0] = ((rewards * mask)[:, i : i + horizon, 0] * gamma_tensor[:horizon]).sum(dim=1)
            indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
            n_targets_terminated = th.gather(target_max_qvals * (1 - terminated), dim=1, index=steps + indices - 1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        td_error = chosen_action_qvals - targets.detach()
        q_mask = mask.expand_as(td_error)
        masked_td_error = td_error * q_mask
        q_loss = (masked_td_error ** 2).sum() / q_mask.sum()

        if self.disable_belief_loss:
            zero = rewards.new_tensor(0.0)
            hidden_belief_loss = zero
            reappear_belief_loss = zero
            visible_belief_loss = zero
            belief_loss = zero
            raw_nll_mean = zero
            belief_logvar_mean = zero
            belief_unseen_frac = zero
            belief_weight = 0.0
            loss = q_loss
        else:
            hidden_belief_loss = hidden_loss_sum / hidden_denom.clamp(min=1.0)
            reappear_belief_loss = reappear_loss_sum / reappear_denom.clamp(min=1.0)
            visible_belief_loss = visible_loss_sum / visible_denom.clamp(min=1.0)
            belief_loss = (
                hidden_belief_loss
                + self.belief_reappear_coef * reappear_belief_loss
                + self.belief_visible_coef * visible_belief_loss
            )
            raw_nll_mean = raw_nll_sum / raw_nll_count.clamp(min=1.0)
            belief_logvar_mean = belief_logvar_sum / belief_logvar_count.clamp(min=1.0)
            belief_unseen_frac = belief_mask_sum / belief_mask_count.clamp(min=1.0)
            belief_weight = self.belief_loss_coef * min(1.0, float(t_env) / float(max(1, self.belief_warmup_t)))
            loss = q_loss + belief_weight * belief_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("belief_raw_nll", raw_nll_mean.item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_hidden_loss", hidden_belief_loss.item(), t_env)
            self.logger.log_stat("belief_reappear_loss", reappear_belief_loss.item(), t_env)
            self.logger.log_stat("belief_visible_loss", visible_belief_loss.item(), t_env)
            self.logger.log_stat("belief_reappear_coef", self.belief_reappear_coef, t_env)
            self.logger.log_stat("belief_visible_coef", self.belief_visible_coef, t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_q_alpha", belief_q_alpha, t_env)
            self.logger.log_stat("belief_logvar_mean", belief_logvar_mean.item(), t_env)
            self.logger.log_stat("belief_unseen_frac", belief_unseen_frac.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = q_mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * q_mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * q_mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
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
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
