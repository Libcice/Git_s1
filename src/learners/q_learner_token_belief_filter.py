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
        self.belief_nll_clip = getattr(self.args, "belief_nll_clip", 2.0)
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
        nll = 0.5 * ((target - mu).pow(2) * th.exp(-logvar) + logvar)
        nll = nll.mean(dim=-1)
        return nll.clamp(min=0.0, max=self.belief_nll_clip)

    def _masked_mean_loss(self, values, mask):
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom

    def _stack_info(self, infos, key):
        return th.stack([info[key] for info in infos], dim=1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        infos = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            infos.append(self.mac.get_belief_filter_info())
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

        enemy_state = self._enemy_state(batch)
        state_target = enemy_state.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1, -1)
        enemy_visible = self._enemy_visible(batch)
        alive = (state_target[..., 0] > 0).float()
        time_agent_mask = mask.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1).squeeze(-1)

        belief_mu = self._stack_info(infos, "belief_mu")
        belief_logvar = self._stack_info(infos, "belief_logvar")
        future_mu = self._stack_info(infos, "future_mu")
        future_logvar = self._stack_info(infos, "future_logvar")
        q_context = self._stack_info(infos, "q_context")
        belief_delta_q = self._stack_info(infos, "belief_delta_q")
        belief_gate = self._stack_info(infos, "belief_gate")
        base_q = self._stack_info(infos, "base_q")
        correction_alpha = self._stack_info(infos, "correction_alpha")
        belief_confidence = self._stack_info(infos, "belief_confidence")

        hidden_mask = (1.0 - enemy_visible) * alive
        belief_mask = hidden_mask[:, :-1] * time_agent_mask.unsqueeze(-1)
        belief_nll = self._gaussian_nll(state_target[:, :-1], belief_mu[:, :-1], belief_logvar[:, :-1])
        belief_loss = self._masked_mean_loss(belief_nll, belief_mask)
        visible_mask = enemy_visible[:, :-1] * alive[:, :-1] * time_agent_mask.unsqueeze(-1)
        visible_loss = self._masked_mean_loss(belief_nll, visible_mask)

        future_target = state_target[:, 1:]
        future_hidden_mask = (1.0 - enemy_visible[:, 1:]) * alive[:, 1:]
        future_valid_mask = time_agent_mask * (1.0 - terminated.squeeze(-1)).unsqueeze(-1)
        future_mask = future_hidden_mask * future_valid_mask.unsqueeze(-1)
        future_nll = self._gaussian_nll(future_target, future_mu[:, :-1], future_logvar[:, :-1])
        future_loss = self._masked_mean_loss(future_nll, future_mask)

        with th.no_grad():
            flat_q_context = q_context[:, :-1].reshape(-1, q_context.size(-1))
            flat_enemy_state = state_target[:, :-1].reshape(-1, self.args.enemy_num, self.enemy_state_feat_dim)
            flat_hidden_mask = hidden_mask[:, :-1].reshape(-1, self.args.enemy_num)
            oracle_delta_q, oracle_gate, _, _ = self.target_mac.agent.compute_oracle_delta_q(
                flat_q_context,
                flat_enemy_state,
                flat_hidden_mask,
            )
            oracle_contrib = oracle_delta_q * oracle_gate
            oracle_contrib = oracle_contrib.view(
                batch.batch_size,
                batch.max_seq_length - 1,
                self.args.n_agents,
                self.args.n_actions,
            ).clamp(min=-self.belief_delta_q_clip, max=self.belief_delta_q_clip)

        student_contrib = belief_delta_q[:, :-1] * belief_gate[:, :-1]
        hidden_present = hidden_mask[:, :-1].sum(dim=-1).gt(0).float()
        delta_mask = (
            time_agent_mask.unsqueeze(-1)
            * hidden_present.unsqueeze(-1)
            * avail_actions[:, :-1].float()
        )
        delta_error = F.smooth_l1_loss(student_contrib, oracle_contrib, reduction="none")
        delta_q_loss = self._masked_mean_loss(delta_error, delta_mask)

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
            self.logger.log_stat("belief_visible_loss", visible_loss.item(), t_env)
            self.logger.log_stat("belief_future_loss", future_loss.item(), t_env)
            self.logger.log_stat("belief_delta_q_loss", delta_q_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_visible_weight", visible_weight, t_env)
            self.logger.log_stat("belief_future_weight", future_weight, t_env)
            self.logger.log_stat("belief_delta_q_weight", delta_q_weight, t_env)
            self.logger.log_stat("belief_aux_q_ratio", aux_q_ratio.item(), t_env)
            self.logger.log_stat("belief_hidden_frac", belief_mask.mean().item(), t_env)
            self.logger.log_stat("belief_visible_frac", visible_mask.mean().item(), t_env)
            self.logger.log_stat("belief_future_hidden_frac", future_mask.mean().item(), t_env)
            self.logger.log_stat("belief_gate_mean", belief_gate[:, :-1].mean().item(), t_env)
            self.logger.log_stat("belief_delta_abs_mean", student_contrib.detach().abs().mean().item(), t_env)
            self.logger.log_stat("belief_oracle_delta_abs_mean", oracle_contrib.detach().abs().mean().item(), t_env)
            self.logger.log_stat("belief_confidence_mean", belief_confidence[:, :-1].mean().item(), t_env)
            self.logger.log_stat("belief_correction_alpha_mean", correction_alpha[:, :-1].mean().item(), t_env)
            self.logger.log_stat("base_q_abs_mean", base_q[:, :-1].detach().abs().mean().item(), t_env)
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
