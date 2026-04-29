import copy

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerTokenBeliefFilter:
    """QMIX learner for token belief-filter agents.

    Auxiliary losses are deliberately conservative. They supervise hidden
    current state, one-step future hidden state, and the action-conditioned
    belief delta without leaking privileged state into the online Q path.
    """

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

        self.target_mac = copy.deepcopy(mac)

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
        self.enemy_obs_feat_dim = getattr(self.args, "enemy_obs_feat_dim", 4 + self.unit_type_bits + health_bits)

        self.belief_loss_coef = getattr(self.args, "belief_loss_coef", 0.005)
        self.belief_visible_coef = getattr(self.args, "belief_visible_coef", 0.001)
        self.belief_future_coef = getattr(self.args, "belief_future_coef", 0.005)
        self.belief_delta_q_coef = getattr(self.args, "belief_delta_q_coef", 0.005)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 300000)
        self.belief_visible_warmup_t = getattr(self.args, "belief_visible_warmup_t", self.belief_warmup_t)
        self.belief_future_warmup_t = getattr(self.args, "belief_future_warmup_t", self.belief_warmup_t)
        self.belief_delta_warmup_t = getattr(self.args, "belief_delta_warmup_t", 600000)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -3.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_nll_error_scale = getattr(self.args, "belief_nll_error_scale", 5.0)
        self.belief_nll_clip = getattr(self.args, "belief_nll_clip", 5.0)
        self.belief_delta_q_clip = getattr(self.args, "belief_delta_q_clip", 1.0)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _warmup(self, coef, t_env, warmup_t):
        return coef * min(1.0, float(t_env) / float(max(1, warmup_t)))

    def _enemy_state(self, batch):
        enemy_state = batch["state"][:, :, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
        return enemy_state.view(
            batch.batch_size,
            batch.max_seq_length,
            self.args.enemy_num,
            self.enemy_state_feat_dim,
        )

    def _enemy_visible(self, batch):
        start = self.move_dim
        end = start + self.args.enemy_num * self.enemy_obs_feat_dim
        enemy_obs = batch["obs"][:, :, :, start:end]
        enemy_obs = enemy_obs.view(
            batch.batch_size,
            batch.max_seq_length,
            self.args.n_agents,
            self.args.enemy_num,
            self.enemy_obs_feat_dim,
        )
        return (enemy_obs.abs().sum(dim=-1) > 0).float()

    def _gaussian_nll(self, target, mu, logvar):
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        # The state features are small/normalised, so unscaled Gaussian NLL can
        # collapse to zero after the log-variance term is shifted/clipped. Scale
        # only the prediction error while keeping logvar as the uncertainty term.
        scaled_diff = (target - mu) * self.belief_nll_error_scale
        nll = 0.5 * (scaled_diff.pow(2) * th.exp(-logvar) + logvar)
        nll = nll.mean(dim=-1)
        # Shift by the minimum possible logvar contribution instead of clamping
        # at zero, so near-correct predictions still keep meaningful gradients.
        nll = nll - 0.5 * self.belief_logvar_min
        return nll.clamp(max=self.belief_nll_clip)

    def _masked_mean_loss(self, values, mask):
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        enemy_state = self._enemy_state(batch)
        state_target = enemy_state.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1, -1)
        enemy_visible = self._enemy_visible(batch)
        alive = (state_target[..., 0] > 0).float()

        mac_out = []
        belief_loss_sum = rewards.new_tensor(0.0)
        belief_loss_denom = rewards.new_tensor(0.0)
        visible_loss_sum = rewards.new_tensor(0.0)
        visible_loss_denom = rewards.new_tensor(0.0)
        future_loss_sum = rewards.new_tensor(0.0)
        future_loss_denom = rewards.new_tensor(0.0)
        delta_q_loss_sum = rewards.new_tensor(0.0)
        delta_q_loss_denom = rewards.new_tensor(0.0)
        belief_abs_error_sum = rewards.new_tensor(0.0)
        belief_abs_error_denom = rewards.new_tensor(0.0)
        future_abs_error_sum = rewards.new_tensor(0.0)
        future_abs_error_denom = rewards.new_tensor(0.0)
        hidden_frac_sum = rewards.new_tensor(0.0)
        visible_frac_sum = rewards.new_tensor(0.0)
        future_hidden_frac_sum = rewards.new_tensor(0.0)
        gate_sum = rewards.new_tensor(0.0)
        delta_abs_sum = rewards.new_tensor(0.0)
        oracle_delta_abs_sum = rewards.new_tensor(0.0)
        confidence_sum = rewards.new_tensor(0.0)
        correction_alpha_sum = rewards.new_tensor(0.0)
        base_q_abs_sum = rewards.new_tensor(0.0)
        metric_count = rewards.new_tensor(0.0)

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            info = self.mac.get_belief_filter_info()
            if t < batch.max_seq_length - 1:
                valid_agent_mask_t = mask[:, t].expand(-1, self.args.n_agents)
                target_t = state_target[:, t]
                visible_t = enemy_visible[:, t]
                alive_t = alive[:, t]
                hidden_mask_t = (1.0 - visible_t) * alive_t
                belief_mask_t = hidden_mask_t * valid_agent_mask_t.unsqueeze(-1)
                belief_nll_t = self._gaussian_nll(target_t, info["belief_mu"], info["belief_logvar"])
                belief_loss_sum = belief_loss_sum + (belief_nll_t * belief_mask_t).sum()
                belief_loss_denom = belief_loss_denom + belief_mask_t.sum()
                belief_abs_error_t = (target_t - info["belief_mu"]).abs().mean(dim=-1)
                belief_abs_error_sum = belief_abs_error_sum + (belief_abs_error_t * belief_mask_t).sum()
                belief_abs_error_denom = belief_abs_error_denom + belief_mask_t.sum()

                visible_mask_t = visible_t * alive_t * valid_agent_mask_t.unsqueeze(-1)
                visible_loss_sum = visible_loss_sum + (belief_nll_t * visible_mask_t).sum()
                visible_loss_denom = visible_loss_denom + visible_mask_t.sum()

                future_target_t = state_target[:, t + 1]
                future_hidden_mask_t = (1.0 - enemy_visible[:, t + 1]) * alive[:, t + 1]
                future_valid_t = valid_agent_mask_t * (1.0 - terminated[:, t, 0]).unsqueeze(-1)
                future_mask_t = future_hidden_mask_t * future_valid_t.unsqueeze(-1)
                future_nll_t = self._gaussian_nll(
                    future_target_t,
                    info["future_mu"],
                    info["future_logvar"],
                )
                future_loss_sum = future_loss_sum + (future_nll_t * future_mask_t).sum()
                future_loss_denom = future_loss_denom + future_mask_t.sum()
                future_abs_error_t = (future_target_t - info["future_mu"]).abs().mean(dim=-1)
                future_abs_error_sum = future_abs_error_sum + (future_abs_error_t * future_mask_t).sum()
                future_abs_error_denom = future_abs_error_denom + future_mask_t.sum()

                with th.no_grad():
                    flat_q_context = info["q_context"].reshape(-1, info["q_context"].size(-1))
                    flat_enemy_state = target_t.reshape(-1, self.args.enemy_num, self.enemy_state_feat_dim)
                    flat_hidden_mask = hidden_mask_t.reshape(-1, self.args.enemy_num)
                    oracle_delta_q, oracle_gate, _, _ = self.target_mac.agent.compute_oracle_delta_q(
                        flat_q_context,
                        flat_enemy_state,
                        flat_hidden_mask,
                    )
                    oracle_contrib_t = (oracle_delta_q * oracle_gate).view(
                        batch.batch_size,
                        self.args.n_agents,
                        self.args.n_actions,
                    ).clamp(min=-self.belief_delta_q_clip, max=self.belief_delta_q_clip)

                student_contrib_t = info["belief_delta_q"] * info["belief_gate"]
                hidden_present_t = hidden_mask_t.sum(dim=-1).gt(0).float()
                delta_mask_t = (
                    valid_agent_mask_t.unsqueeze(-1)
                    * hidden_present_t.unsqueeze(-1)
                    * avail_actions[:, t].float()
                )
                delta_error_t = F.smooth_l1_loss(student_contrib_t, oracle_contrib_t, reduction="none")
                delta_q_loss_sum = delta_q_loss_sum + (delta_error_t * delta_mask_t).sum()
                delta_q_loss_denom = delta_q_loss_denom + delta_mask_t.sum()

                hidden_frac_sum = hidden_frac_sum + belief_mask_t.mean()
                visible_frac_sum = visible_frac_sum + visible_mask_t.mean()
                future_hidden_frac_sum = future_hidden_frac_sum + future_mask_t.mean()
                gate_sum = gate_sum + info["belief_gate"].detach().mean()
                delta_abs_sum = delta_abs_sum + student_contrib_t.detach().abs().mean()
                oracle_delta_abs_sum = oracle_delta_abs_sum + oracle_contrib_t.detach().abs().mean()
                confidence_sum = confidence_sum + info["belief_confidence"].mean()
                correction_alpha_sum = correction_alpha_sum + info["correction_alpha"].mean()
                base_q_abs_sum = base_q_abs_sum + info["base_q"].abs().mean()
                metric_count = metric_count + 1.0
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
                n_rewards[:, i, 0] = (
                    (rewards * mask)[:, i:i + horizon, 0] * gamma_tensor[:horizon]
                ).sum(dim=1)
            indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
            n_targets_terminated = th.gather(
                target_max_qvals * (1 - terminated),
                dim=1,
                index=steps + indices - 1,
            )
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        td_error = chosen_action_qvals - targets.detach()
        q_mask = mask.expand_as(td_error)
        masked_td_error = td_error * q_mask
        q_loss = (masked_td_error ** 2).sum() / q_mask.sum()

        belief_loss = belief_loss_sum / belief_loss_denom.clamp(min=1.0)
        visible_loss = visible_loss_sum / visible_loss_denom.clamp(min=1.0)
        future_loss = future_loss_sum / future_loss_denom.clamp(min=1.0)
        delta_q_loss = delta_q_loss_sum / delta_q_loss_denom.clamp(min=1.0)
        belief_abs_error_mean = belief_abs_error_sum / belief_abs_error_denom.clamp(min=1.0)
        future_abs_error_mean = future_abs_error_sum / future_abs_error_denom.clamp(min=1.0)
        metric_denom = metric_count.clamp(min=1.0)
        belief_hidden_frac = hidden_frac_sum / metric_denom
        belief_visible_frac = visible_frac_sum / metric_denom
        belief_future_hidden_frac = future_hidden_frac_sum / metric_denom
        belief_gate_mean = gate_sum / metric_denom
        belief_delta_abs_mean = delta_abs_sum / metric_denom
        belief_oracle_delta_abs_mean = oracle_delta_abs_sum / metric_denom
        belief_confidence_mean = confidence_sum / metric_denom
        belief_correction_alpha_mean = correction_alpha_sum / metric_denom
        base_q_abs_mean = base_q_abs_sum / metric_denom

        belief_weight = self._warmup(self.belief_loss_coef, t_env, self.belief_warmup_t)
        visible_weight = self._warmup(self.belief_visible_coef, t_env, self.belief_visible_warmup_t)
        future_weight = self._warmup(self.belief_future_coef, t_env, self.belief_future_warmup_t)
        delta_q_weight = self._warmup(self.belief_delta_q_coef, t_env, self.belief_delta_warmup_t)
        aux_weighted = (
            belief_weight * belief_loss
            + visible_weight * visible_loss
            + future_weight * future_loss
            + delta_q_weight * delta_q_loss
        )
        loss = q_loss + aux_weighted

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            q_loss_detached = q_loss.detach().abs().clamp(min=1.0e-6)
            aux_q_ratio = aux_weighted.detach() / q_loss_detached
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_abs_error", belief_abs_error_mean.item(), t_env)
            self.logger.log_stat("belief_visible_loss", visible_loss.item(), t_env)
            self.logger.log_stat("belief_future_loss", future_loss.item(), t_env)
            self.logger.log_stat("belief_future_abs_error", future_abs_error_mean.item(), t_env)
            self.logger.log_stat("belief_delta_q_loss", delta_q_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_visible_weight", visible_weight, t_env)
            self.logger.log_stat("belief_future_weight", future_weight, t_env)
            self.logger.log_stat("belief_delta_q_weight", delta_q_weight, t_env)
            self.logger.log_stat("belief_aux_q_ratio", aux_q_ratio.item(), t_env)
            self.logger.log_stat("belief_hidden_frac", belief_hidden_frac.item(), t_env)
            self.logger.log_stat("belief_visible_frac", belief_visible_frac.item(), t_env)
            self.logger.log_stat("belief_future_hidden_frac", belief_future_hidden_frac.item(), t_env)
            self.logger.log_stat("belief_gate_mean", belief_gate_mean.item(), t_env)
            self.logger.log_stat("belief_delta_abs_mean", belief_delta_abs_mean.item(), t_env)
            self.logger.log_stat("belief_oracle_delta_abs_mean", belief_oracle_delta_abs_mean.item(), t_env)
            self.logger.log_stat("belief_confidence_mean", belief_confidence_mean.item(), t_env)
            self.logger.log_stat("belief_correction_alpha_mean", belief_correction_alpha_mean.item(), t_env)
            self.logger.log_stat("base_q_abs_mean", base_q_abs_mean.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
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
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
