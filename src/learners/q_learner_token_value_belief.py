import copy
import os

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerTokenValueBelief:
    """QMIX learner for local counterfactual value-belief slots."""

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

        self.hidden_dim = getattr(self.args, "transformer_hidden_dim", self.args.rnn_hidden_dim)
        self.belief_q_warmup_t = getattr(self.args, "belief_q_warmup_t", 300000)
        self.belief_delta_warmup_t = getattr(self.args, "belief_delta_warmup_t", 300000)
        self.belief_delta_total_coef = getattr(self.args, "belief_delta_total_coef", 0.25)
        self.belief_delta_slot_coef = getattr(self.args, "belief_delta_slot_coef", 0.10)
        self.belief_rank_coef = getattr(self.args, "belief_rank_coef", 0.05)
        self.belief_aux_coef = getattr(self.args, "belief_aux_coef", 0.02)
        self.teacher_q_coef = getattr(self.args, "belief_teacher_q_coef", 0.10)

        self.use_delta_total = self.belief_delta_total_coef > 0.0
        self.use_delta_slot = self.belief_delta_slot_coef > 0.0
        self.use_rank = self.belief_rank_coef > 0.0
        self.use_aux = self.belief_aux_coef > 0.0
        self.need_teacher_targets = self.use_delta_total or self.use_delta_slot or self.use_rank
        if self.need_teacher_targets and self.teacher_q_coef <= 0.0:
            self.teacher_q_coef = 0.10
        self.use_teacher_q = self.teacher_q_coef > 0.0
        self.use_belief_q = self.need_teacher_targets or self.use_teacher_q
        self.disable_value_belief = (not self.use_belief_q) and (not self.use_aux)

        teacher_enemy_input_dim = self.enemy_state_feat_dim + self.ally_state_feat_dim * 2 + 1
        self.teacher_enemy_encoder = th.nn.Sequential(
            th.nn.Linear(teacher_enemy_input_dim, self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        teacher_query_input_dim = self.ally_state_feat_dim * 2 + self.enemy_state_feat_dim + 2
        self.teacher_query_encoder = th.nn.Sequential(
            th.nn.Linear(teacher_query_input_dim, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.teacher_delta_q_head = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.args.n_actions),
        )
        self.params += list(self.teacher_enemy_encoder.parameters())
        self.params += list(self.teacher_query_encoder.parameters())
        self.params += list(self.teacher_delta_q_head.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.use_belief_q:
            belief_q_alpha = min(1.0, float(t_env) / float(max(1, self.belief_q_warmup_t)))
        else:
            belief_q_alpha = 0.0
        if hasattr(self.mac, "set_belief_q_alpha"):
            self.mac.set_belief_q_alpha(belief_q_alpha)
        if hasattr(self.target_mac, "set_belief_q_alpha"):
            self.target_mac.set_belief_q_alpha(belief_q_alpha)

        warm = min(1.0, float(t_env) / float(max(1, self.belief_delta_warmup_t)))
        mac_out = []
        teacher_chosen_q_list = []
        teacher_global_mask_list = []

        delta_total_sum = rewards.new_tensor(0.0)
        delta_total_denom = rewards.new_tensor(0.0)
        delta_slot_sum = rewards.new_tensor(0.0)
        delta_slot_denom = rewards.new_tensor(0.0)
        rank_loss_sum = rewards.new_tensor(0.0)
        rank_denom = rewards.new_tensor(0.0)
        aux_loss_sum = rewards.new_tensor(0.0)
        aux_denom = rewards.new_tensor(0.0)
        delta_abs_sum = rewards.new_tensor(0.0)
        delta_abs_count = rewards.new_tensor(0.0)
        q_visible_sum = rewards.new_tensor(0.0)
        q_visible_count = rewards.new_tensor(0.0)
        alive_prob_sum = rewards.new_tensor(0.0)
        alive_prob_count = rewards.new_tensor(0.0)
        hidden_frac_sum = rewards.new_tensor(0.0)
        hidden_frac_count = rewards.new_tensor(0.0)
        rel_hidden_sum = rewards.new_tensor(0.0)
        rel_hidden_count = rewards.new_tensor(0.0)
        teacher_delta_abs_sum = rewards.new_tensor(0.0)
        teacher_delta_abs_count = rewards.new_tensor(0.0)
        delta_ratio_sum = rewards.new_tensor(0.0)
        delta_ratio_count = rewards.new_tensor(0.0)
        slot_norm_sum = rewards.new_tensor(0.0)
        slot_norm_count = rewards.new_tensor(0.0)
        enemy_weight_sum = rewards.new_tensor(0.0)
        enemy_weight_count = rewards.new_tensor(0.0)

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

            if self.disable_value_belief:
                continue

            q_visible_t = self.mac.get_q_visible()
            delta_q_hidden_t = self.mac.get_delta_q_hidden()
            delta_q_per_enemy_t = self.mac.get_delta_q_per_enemy()
            rel_logits_t = self.mac.get_rel_logits()
            alive_logits_t, hp_preds_t = self.mac.get_aux_stats()
            belief_slots_t = self.mac.get_belief()
            enemy_weight_t = self.mac.get_enemy_weight()

            enemy_state_t = batch["state"][:, t, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
            enemy_state_t = enemy_state_t.reshape(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
            state_target_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)

            enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
            enemy_obs_t = enemy_obs_t.reshape(batch.batch_size, self.args.n_agents, self.args.enemy_num, self.enemy_obs_feat_dim)
            enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
            hidden_mask_t = 1.0 - enemy_visible_t
            alive_target_t = (state_target_t[..., 0] > 0).float()
            hidden_alive_mask_t = hidden_mask_t * alive_target_t

            valid_agent_mask_t = None
            if t < batch.max_seq_length - 1:
                valid_step_mask_t = mask[:, t].reshape(batch.batch_size, -1)[:, 0]
                valid_agent_mask_t = valid_step_mask_t.unsqueeze(1).expand(-1, self.args.n_agents)

            teacher_delta_q_t = None
            teacher_delta_q_hidden_t = None
            if self.use_belief_q:
                ally_state_t = batch["state"][:, t, :self.ally_state_dim]
                ally_state_t = ally_state_t.reshape(batch.batch_size, self.args.n_agents, self.ally_state_feat_dim)
                own_state_t = ally_state_t
                ally_mean_agent_t = ally_state_t.mean(dim=1, keepdim=True).expand(-1, self.args.n_agents, -1)
                visible_enemy_mean_t = self._masked_mean(
                    state_target_t.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num, self.enemy_state_feat_dim),
                    enemy_visible_t.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num),
                ).reshape(batch.batch_size, self.args.n_agents, self.enemy_state_feat_dim)
                visible_frac_t = enemy_visible_t.mean(dim=-1, keepdim=True)
                hidden_frac_t = hidden_mask_t.mean(dim=-1, keepdim=True)
                teacher_query_t = self.teacher_query_encoder(
                    th.cat(
                        [own_state_t, ally_mean_agent_t, visible_enemy_mean_t, visible_frac_t, hidden_frac_t],
                        dim=-1,
                    )
                )
                teacher_query_expand_t = teacher_query_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                own_state_expand_t = own_state_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                ally_mean_expand_t = ally_mean_agent_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                teacher_enemy_input_t = th.cat(
                    [
                        state_target_t,
                        own_state_expand_t,
                        ally_mean_expand_t,
                        enemy_visible_t.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                teacher_enemy_feat_t = self.teacher_enemy_encoder(teacher_enemy_input_t)
                teacher_delta_q_t = self.teacher_delta_q_head(
                    th.cat([teacher_query_expand_t, teacher_enemy_feat_t], dim=-1)
                )
                teacher_delta_q_hidden_t = (hidden_alive_mask_t.unsqueeze(-1) * teacher_delta_q_t).sum(dim=2)

            if t < batch.max_seq_length - 1:
                if self.use_delta_total:
                    delta_total_error_t = F.smooth_l1_loss(
                        delta_q_hidden_t,
                        teacher_delta_q_hidden_t.detach(),
                        reduction="none",
                    ).mean(dim=-1)
                    delta_total_sum = delta_total_sum + (delta_total_error_t * valid_agent_mask_t).sum()
                    delta_total_denom = delta_total_denom + valid_agent_mask_t.sum()

                if self.use_delta_slot:
                    delta_slot_error_t = F.smooth_l1_loss(
                        delta_q_per_enemy_t,
                        teacher_delta_q_t.detach(),
                        reduction="none",
                    ).mean(dim=-1)
                    delta_slot_mask_t = hidden_alive_mask_t * valid_agent_mask_t.unsqueeze(-1)
                    delta_slot_sum = delta_slot_sum + (delta_slot_error_t * delta_slot_mask_t).sum()
                    delta_slot_denom = delta_slot_denom + delta_slot_mask_t.sum()

                if self.use_rank:
                    teacher_score_t = teacher_delta_q_t.detach().abs().mean(dim=-1)
                    valid_rank_mask_t = hidden_alive_mask_t > 0
                    valid_rows_t = (valid_rank_mask_t.sum(dim=-1) > 0) & (valid_agent_mask_t > 0)
                    if valid_rows_t.any():
                        teacher_score_rows = teacher_score_t[valid_rows_t].masked_fill(~valid_rank_mask_t[valid_rows_t], -1e9)
                        rel_logit_rows = rel_logits_t[valid_rows_t].masked_fill(~valid_rank_mask_t[valid_rows_t], -1e9)
                        teacher_prob_rows = th.softmax(teacher_score_rows, dim=-1)
                        student_log_prob_rows = th.log_softmax(rel_logit_rows, dim=-1)
                        rank_loss_sum = rank_loss_sum - (teacher_prob_rows * student_log_prob_rows).sum()
                        rank_denom = rank_denom + teacher_prob_rows.new_tensor(float(valid_rows_t.sum().item()))

                if self.use_aux:
                    hp_target_t = state_target_t[..., 0].clamp(min=0.0)
                    alive_loss_t = F.binary_cross_entropy_with_logits(
                        alive_logits_t,
                        alive_target_t,
                        reduction="none",
                    )
                    hp_loss_t = F.smooth_l1_loss(
                        hp_preds_t,
                        hp_target_t,
                        reduction="none",
                    )
                    aux_error_t = alive_loss_t + hp_loss_t
                    aux_mask_t = valid_agent_mask_t.unsqueeze(-1).expand(-1, -1, self.args.enemy_num)
                    aux_loss_sum = aux_loss_sum + (aux_error_t * aux_mask_t).sum()
                    aux_denom = aux_denom + aux_mask_t.sum()

                delta_abs_sum = delta_abs_sum + (delta_q_hidden_t.abs().mean(dim=-1) * valid_agent_mask_t).sum()
                delta_abs_count = delta_abs_count + valid_agent_mask_t.sum()
                q_visible_sum = q_visible_sum + (q_visible_t.mean(dim=-1) * valid_agent_mask_t).sum()
                q_visible_count = q_visible_count + valid_agent_mask_t.sum()
                alive_prob_sum = alive_prob_sum + (th.sigmoid(alive_logits_t).mean(dim=-1) * valid_agent_mask_t).sum()
                alive_prob_count = alive_prob_count + valid_agent_mask_t.sum()
                hidden_frac_sum = hidden_frac_sum + (hidden_mask_t.mean(dim=-1) * valid_agent_mask_t).sum()
                hidden_frac_count = hidden_frac_count + valid_agent_mask_t.sum()
                delta_ratio_t = delta_q_hidden_t.abs().mean(dim=-1) / (q_visible_t.abs().mean(dim=-1) + 1e-6)
                delta_ratio_sum = delta_ratio_sum + (delta_ratio_t * valid_agent_mask_t).sum()
                delta_ratio_count = delta_ratio_count + valid_agent_mask_t.sum()
                slot_norm_t = belief_slots_t.norm(dim=-1).mean(dim=-1)
                slot_norm_sum = slot_norm_sum + (slot_norm_t * valid_agent_mask_t).sum()
                slot_norm_count = slot_norm_count + valid_agent_mask_t.sum()
                hidden_valid_mask_t = hidden_mask_t * valid_agent_mask_t.unsqueeze(-1)
                rel_hidden_sum = rel_hidden_sum + (th.sigmoid(rel_logits_t) * hidden_valid_mask_t).sum()
                rel_hidden_count = rel_hidden_count + hidden_valid_mask_t.sum()
                enemy_weight_sum = enemy_weight_sum + (enemy_weight_t * hidden_valid_mask_t).sum()
                enemy_weight_count = enemy_weight_count + hidden_valid_mask_t.sum()
                if self.use_belief_q:
                    teacher_delta_abs_sum = teacher_delta_abs_sum + (
                        teacher_delta_q_hidden_t.abs().mean(dim=-1) * valid_agent_mask_t
                    ).sum()
                    teacher_delta_abs_count = teacher_delta_abs_count + valid_agent_mask_t.sum()

            if self.use_teacher_q:
                teacher_full_q_t = q_visible_t.detach() + teacher_delta_q_hidden_t
                if t < batch.max_seq_length - 1:
                    teacher_chosen_q_t = th.gather(teacher_full_q_t, dim=2, index=actions[:, t]).squeeze(2)
                    teacher_chosen_q_list.append(teacher_chosen_q_t)
                    teacher_global_mask_list.append((hidden_alive_mask_t.sum(dim=-1) > 0).float())

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
                n_rewards[:, i, 0] = ((rewards * mask)[:, i:i + horizon, 0] * gamma_tensor[:horizon]).sum(dim=1)
            indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
            n_targets_terminated = th.gather(target_max_qvals * (1 - terminated), dim=1, index=steps + indices - 1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        td_error = chosen_action_qvals - targets.detach()
        q_mask = mask.expand_as(td_error)
        masked_td_error = td_error * q_mask
        q_loss = (masked_td_error ** 2).sum() / q_mask.sum()

        delta_total_loss = delta_total_sum / delta_total_denom.clamp(min=1.0)
        delta_slot_loss = delta_slot_sum / delta_slot_denom.clamp(min=1.0)
        rank_loss = rank_loss_sum / rank_denom.clamp(min=1.0)
        aux_loss = aux_loss_sum / aux_denom.clamp(min=1.0)
        teacher_q_loss = rewards.new_tensor(0.0)
        teacher_q_weight = 0.0

        if self.use_teacher_q:
            teacher_chosen_action_qvals = th.stack(teacher_chosen_q_list, dim=1)
            if self.mixer is not None:
                teacher_chosen_action_qvals = self.mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])
            teacher_td_error = teacher_chosen_action_qvals - targets.detach()
            teacher_masked_td_error = teacher_td_error * q_mask
            teacher_global_mask = th.stack(teacher_global_mask_list, dim=1)
            if self.mixer is not None:
                teacher_global_mask = (teacher_global_mask.sum(dim=-1, keepdim=True) > 0).float()
            teacher_q_loss = ((teacher_masked_td_error ** 2) * teacher_global_mask).sum() / teacher_global_mask.sum().clamp(min=1.0)
            teacher_q_weight = self.teacher_q_coef * warm

        delta_total_weight = self.belief_delta_total_coef * warm
        delta_slot_weight = self.belief_delta_slot_coef * warm
        rank_weight = self.belief_rank_coef * warm
        aux_weight = self.belief_aux_coef * warm
        loss = (
            q_loss
            + teacher_q_weight * teacher_q_loss
            + delta_total_weight * delta_total_loss
            + delta_slot_weight * delta_slot_loss
            + rank_weight * rank_loss
            + aux_weight * aux_loss
        )

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
            self.logger.log_stat("belief_q_alpha", belief_q_alpha, t_env)
            self.logger.log_stat("belief_delta_total_loss", delta_total_loss.item(), t_env)
            self.logger.log_stat("belief_delta_total_weight", delta_total_weight, t_env)
            self.logger.log_stat("belief_delta_slot_loss", delta_slot_loss.item(), t_env)
            self.logger.log_stat("belief_delta_slot_weight", delta_slot_weight, t_env)
            self.logger.log_stat("belief_rank_loss", rank_loss.item(), t_env)
            self.logger.log_stat("belief_rank_weight", rank_weight, t_env)
            self.logger.log_stat("belief_aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("belief_aux_weight", aux_weight, t_env)
            self.logger.log_stat("belief_teacher_q_loss", teacher_q_loss.item(), t_env)
            self.logger.log_stat("belief_teacher_q_weight", teacher_q_weight, t_env)
            self.logger.log_stat("belief_delta_abs_mean", (delta_abs_sum / delta_abs_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_q_visible_mean", (q_visible_sum / q_visible_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_alive_prob_mean", (alive_prob_sum / alive_prob_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_hidden_frac", (hidden_frac_sum / hidden_frac_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_rel_hidden_mean", (rel_hidden_sum / rel_hidden_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_enemy_weight_mean", (enemy_weight_sum / enemy_weight_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_teacher_delta_abs_mean", (teacher_delta_abs_sum / teacher_delta_abs_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_delta_ratio", (delta_ratio_sum / delta_ratio_count.clamp(min=1.0)).item(), t_env)
            self.logger.log_stat("belief_slot_norm", (slot_norm_sum / slot_norm_count.clamp(min=1.0)).item(), t_env)
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
        self.teacher_enemy_encoder.cuda()
        self.teacher_query_encoder.cuda()
        self.teacher_delta_q_head.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.teacher_enemy_encoder.state_dict(), "{}/teacher_enemy_encoder.th".format(path))
        th.save(self.teacher_query_encoder.state_dict(), "{}/teacher_query_encoder.th".format(path))
        th.save(self.teacher_delta_q_head.state_dict(), "{}/teacher_delta_q_head.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        enemy_path = "{}/teacher_enemy_encoder.th".format(path)
        if os.path.exists(enemy_path):
            self.teacher_enemy_encoder.load_state_dict(
                th.load(enemy_path, map_location=lambda storage, loc: storage)
            )
        query_path = "{}/teacher_query_encoder.th".format(path)
        if os.path.exists(query_path):
            self.teacher_query_encoder.load_state_dict(
                th.load(query_path, map_location=lambda storage, loc: storage)
            )
        delta_path = "{}/teacher_delta_q_head.th".format(path)
        if os.path.exists(delta_path):
            self.teacher_delta_q_head.load_state_dict(
                th.load(delta_path, map_location=lambda storage, loc: storage)
            )
        opt_state = th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        try:
            self.optimiser.load_state_dict(opt_state)
        except ValueError:
            pass
