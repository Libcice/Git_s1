import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearnerHistoryTokenBelief:
    """QMIX learner with belief supervision on unseen enemy state only."""

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
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        belief_mu_out = []
        belief_logvar_out = []
        belief_u_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            belief_mu_t, belief_logvar_t, belief_u_t = self.mac.get_belief_stats()
            belief_mu_out.append(belief_mu_t)
            belief_logvar_out.append(belief_logvar_t)
            belief_u_out.append(belief_u_t)

        mac_out = th.stack(mac_out, dim=1)
        belief_mu_out = th.stack(belief_mu_out, dim=1)
        belief_logvar_out = th.stack(belief_logvar_out, dim=1)
        belief_u_out = th.stack(belief_u_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
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
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma ** i for i in range(N)], dtype=th.float, device=rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                horizon = min(N, batch.max_seq_length - 1 - i)
                n_rewards[:, i, 0] = ((rewards * mask)[:, i:i + horizon, 0] * gamma_tensor[:horizon]).sum(dim=1)
            indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
            n_targets_terminated = th.gather(target_max_qvals * (1 - terminated), dim=1, index=steps + indices - 1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        td_error = chosen_action_qvals - targets.detach()
        q_mask = mask.expand_as(td_error)
        masked_td_error = td_error * q_mask
        q_loss = (masked_td_error ** 2).sum() / q_mask.sum()

        belief_mu = belief_mu_out[:, :-1]
        belief_logvar = belief_logvar_out[:, :-1].clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        belief_u = belief_u_out[:, :-1]

        enemy_state = batch["state"][:, :-1, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
        enemy_state = enemy_state.view(batch.batch_size, -1, self.args.enemy_num, self.enemy_state_feat_dim)
        state_target = enemy_state.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1, -1)

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim
        enemy_obs = batch["obs"][:, :-1, :, enemy_obs_start:enemy_obs_end]
        enemy_obs = enemy_obs.view(batch.batch_size, -1, self.args.n_agents, self.args.enemy_num, self.enemy_obs_feat_dim)
        unseen_mask = (enemy_obs.abs().sum(dim=-1) == 0).float()
        alive_mask = (state_target[..., 0] > 0).float()
        belief_mask = unseen_mask * alive_mask

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

        # Handle mask dimensions flexibly: mask can be [B, T, 1] or [B, T, 1, 1]
        time_agent_mask = mask.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)

        belief_mask = belief_mask * time_agent_mask
        belief_denom = belief_mask.sum().clamp(min=1.0)
        belief_loss = (nll * belief_mask).sum() / belief_denom
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
            self.logger.log_stat("belief_raw_nll", raw_nll.mean().item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_logvar_mean", belief_logvar.mean().item(), t_env)
            self.logger.log_stat("belief_unseen_frac", belief_mask.mean().item(), t_env)
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
