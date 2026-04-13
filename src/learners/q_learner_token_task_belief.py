import copy

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerTokenTaskBelief:
    """QMIX learner for query-conditioned belief readout over hidden entities."""

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
        self.belief_topk = max(0, int(getattr(self.args, "belief_topk", 2)))
        self.teacher_mode = getattr(self.args, "belief_teacher_mode", "loo_q")
        self.teacher_loo_topk = max(1, int(getattr(self.args, "belief_teacher_loo_topk", 2)))
        self.belief_loss_warmup_t = getattr(self.args, "belief_loss_warmup_t", 300000)
        self.belief_latent_align_coef = getattr(self.args, "belief_latent_align_coef", 0.10)
        self.belief_attn_align_coef = getattr(self.args, "belief_attn_align_coef", 0.05)
        self.belief_context_align_coef = getattr(self.args, "belief_context_align_coef", 0.10)
        self.teacher_q_coef = getattr(self.args, "belief_teacher_q_coef", 0.10)
        self.belief_reappear_coef = getattr(self.args, "belief_reappear_coef", 0.10)
        if self.teacher_mode != "loo_q":
            raise ValueError("Unsupported belief_teacher_mode: {}".format(self.teacher_mode))

        self.use_latent_align = self.belief_latent_align_coef > 0.0
        self.use_attn_align = self.belief_attn_align_coef > 0.0
        self.use_context_align = self.belief_context_align_coef > 0.0
        self.use_teacher_q = self.teacher_q_coef > 0.0
        self.use_reappear = self.belief_reappear_coef > 0.0
        self.use_teacher_targets = (
            self.use_latent_align
            or self.use_attn_align
            or self.use_context_align
            or self.use_teacher_q
        )

        teacher_enemy_input_dim = self.enemy_state_feat_dim + self.ally_state_feat_dim * 2 + 1
        self.teacher_enemy_encoder = th.nn.Sequential(
            th.nn.Linear(teacher_enemy_input_dim, self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.teacher_query_proj = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.teacher_key_proj = th.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.teacher_q_head = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.args.n_actions),
        )
        self.teacher_attn_scale = 1.0 / (float(self.hidden_dim) ** 0.5)
        self.reappear_proj = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim, self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.params += list(self.teacher_enemy_encoder.parameters())
        self.params += list(self.teacher_query_proj.parameters())
        self.params += list(self.teacher_key_proj.parameters())
        self.params += list(self.teacher_q_head.parameters())
        self.params += list(self.reappear_proj.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _masked_topk_softmax(self, logits, mask):
        mask_bool = mask > 0
        masked_logits = logits.masked_fill(~mask_bool, -1e9)

        if 0 < self.belief_topk < logits.size(-1):
            topk_k = min(self.belief_topk, logits.size(-1))
            topk_idx = masked_logits.topk(k=topk_k, dim=-1).indices
            topk_mask = th.zeros_like(mask_bool)
            topk_mask.scatter_(-1, topk_idx, True)
            mask_bool = mask_bool & topk_mask
            masked_logits = logits.masked_fill(~mask_bool, -1e9)

        probs = th.softmax(masked_logits, dim=-1)
        probs = probs * mask_bool.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        empty_rows = mask_bool.float().sum(dim=-1, keepdim=True) <= 0
        probs = th.where(empty_rows, th.zeros_like(probs), probs)
        return probs

    def _normalise_teacher_weights(self, scores, mask):
        masked_scores = scores * mask
        score_sum = masked_scores.sum(dim=-1, keepdim=True)
        mask_sum = mask.sum(dim=-1, keepdim=True)
        uniform = mask / mask_sum.clamp(min=1.0)
        probs = masked_scores / score_sum.clamp(min=1e-6)
        zero_delta_rows = (score_sum <= 1e-6) & (mask_sum > 0)
        probs = th.where(zero_delta_rows, uniform, probs)
        probs = th.where(mask_sum > 0, probs, th.zeros_like(probs))
        value_row_mask = (score_sum.squeeze(-1) > 1e-6).float()
        return probs, value_row_mask

    def _build_value_teacher(self, q_context_t, teacher_enemy_feat_t, hidden_present_mask_t, action_t):
        teacher_query_t = self.teacher_query_proj(q_context_t.detach())
        teacher_keys_t = self.teacher_key_proj(teacher_enemy_feat_t)
        teacher_attn_logits_t = (
            teacher_keys_t * teacher_query_t.unsqueeze(2)
        ).sum(dim=-1) * self.teacher_attn_scale
        teacher_base_attn_t = self._masked_topk_softmax(teacher_attn_logits_t, hidden_present_mask_t)
        teacher_base_context_t = (teacher_base_attn_t.unsqueeze(-1) * teacher_enemy_feat_t).sum(dim=2)
        teacher_full_q_t = self.teacher_q_head(
            th.cat([q_context_t.detach(), teacher_base_context_t], dim=-1)
        )
        teacher_chosen_q_t = th.gather(teacher_full_q_t, dim=2, index=action_t).squeeze(2)

        loo_k = min(self.teacher_loo_topk, self.args.enemy_num)
        candidate_scores_t = teacher_attn_logits_t.masked_fill(hidden_present_mask_t <= 0, -1e9)
        candidate_idx_t = candidate_scores_t.topk(k=loo_k, dim=-1).indices
        candidate_valid_t = th.gather(hidden_present_mask_t, dim=-1, index=candidate_idx_t)
        candidate_onehot_t = F.one_hot(candidate_idx_t, num_classes=self.args.enemy_num).float()
        candidate_onehot_t = candidate_onehot_t * candidate_valid_t.unsqueeze(-1)

        minus_mask_t = hidden_present_mask_t.unsqueeze(2) * (1.0 - candidate_onehot_t)
        minus_attn_t = self._masked_topk_softmax(
            teacher_attn_logits_t.unsqueeze(2).expand(-1, -1, loo_k, -1),
            minus_mask_t,
        )
        minus_context_t = (minus_attn_t.unsqueeze(-1) * teacher_enemy_feat_t.unsqueeze(2)).sum(dim=-2)
        minus_q_t = self.teacher_q_head(
            th.cat(
                [q_context_t.detach().unsqueeze(2).expand(-1, -1, loo_k, -1), minus_context_t],
                dim=-1,
            )
        )
        minus_chosen_q_t = th.gather(
            minus_q_t,
            dim=3,
            index=action_t.unsqueeze(2).expand(-1, -1, loo_k, -1),
        ).squeeze(3)

        delta_candidates_t = (teacher_chosen_q_t.unsqueeze(-1) - minus_chosen_q_t).abs() * candidate_valid_t
        teacher_delta_t = hidden_present_mask_t.new_zeros(hidden_present_mask_t.shape)
        teacher_delta_t.scatter_add_(-1, candidate_idx_t, delta_candidates_t)
        teacher_delta_t = teacher_delta_t * hidden_present_mask_t

        teacher_value_attn_t, value_row_mask_t = self._normalise_teacher_weights(
            teacher_delta_t,
            hidden_present_mask_t,
        )
        teacher_value_context_t = (teacher_value_attn_t.unsqueeze(-1) * teacher_enemy_feat_t).sum(dim=2)
        teacher_eval_count_t = candidate_valid_t.sum(dim=-1)
        return (
            teacher_value_attn_t,
            teacher_value_context_t,
            teacher_full_q_t,
            teacher_chosen_q_t,
            value_row_mask_t,
            teacher_delta_t,
            teacher_eval_count_t,
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        warm = min(1.0, float(t_env) / float(max(1, self.belief_loss_warmup_t)))
        mac_out = []
        teacher_chosen_q_list = []
        teacher_global_mask_list = []

        latent_align_sum = rewards.new_tensor(0.0)
        latent_align_denom = rewards.new_tensor(0.0)
        attn_align_sum = rewards.new_tensor(0.0)
        attn_align_denom = rewards.new_tensor(0.0)
        context_align_sum = rewards.new_tensor(0.0)
        context_align_denom = rewards.new_tensor(0.0)
        reappear_sum = rewards.new_tensor(0.0)
        reappear_denom = rewards.new_tensor(0.0)

        hidden_frac_sum = rewards.new_tensor(0.0)
        hidden_frac_count = rewards.new_tensor(0.0)
        slot_norm_sum = rewards.new_tensor(0.0)
        slot_norm_count = rewards.new_tensor(0.0)
        context_norm_sum = rewards.new_tensor(0.0)
        context_norm_count = rewards.new_tensor(0.0)
        teacher_context_norm_sum = rewards.new_tensor(0.0)
        teacher_context_norm_count = rewards.new_tensor(0.0)
        attn_entropy_sum = rewards.new_tensor(0.0)
        attn_entropy_count = rewards.new_tensor(0.0)
        teacher_attn_entropy_sum = rewards.new_tensor(0.0)
        teacher_attn_entropy_count = rewards.new_tensor(0.0)
        top1_mass_sum = rewards.new_tensor(0.0)
        top1_mass_count = rewards.new_tensor(0.0)
        teacher_top1_mass_sum = rewards.new_tensor(0.0)
        teacher_top1_mass_count = rewards.new_tensor(0.0)
        teacher_delta_mean_sum = rewards.new_tensor(0.0)
        teacher_delta_mean_count = rewards.new_tensor(0.0)
        teacher_delta_max_sum = rewards.new_tensor(0.0)
        teacher_delta_max_count = rewards.new_tensor(0.0)
        teacher_eval_count_sum = rewards.new_tensor(0.0)
        teacher_eval_frac_sum = rewards.new_tensor(0.0)
        teacher_eval_frac_count = rewards.new_tensor(0.0)
        prev_slot_latent_t = None
        prev_enemy_visible_t = None

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

            if t >= batch.max_seq_length - 1:
                continue

            q_context_t = self.mac.get_q_context()
            belief_slots_t = self.mac.get_belief()
            slot_latent_t = self.mac.get_slot_latent()
            belief_attn_t = self.mac.get_belief_attn()
            belief_context_t = self.mac.get_belief_context()
            enemy_ctx_t = self.mac.get_enemy_context()

            enemy_state_t = batch["state"][:, t, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
            enemy_state_t = enemy_state_t.reshape(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
            state_target_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)
            entity_present_t = (state_target_t.abs().sum(dim=-1) > 0).float()

            enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
            enemy_obs_t = enemy_obs_t.reshape(
                batch.batch_size,
                self.args.n_agents,
                self.args.enemy_num,
                self.enemy_obs_feat_dim,
            )
            enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
            hidden_mask_t = 1.0 - enemy_visible_t
            hidden_present_mask_t = hidden_mask_t * entity_present_t

            valid_step_mask_t = mask[:, t].reshape(batch.batch_size, -1)[:, 0]
            valid_agent_mask_t = valid_step_mask_t.unsqueeze(1).expand(-1, self.args.n_agents)

            if self.use_reappear and prev_slot_latent_t is not None:
                reappear_mask_t = enemy_visible_t * (1.0 - prev_enemy_visible_t) * entity_present_t
                reappear_mask_t = reappear_mask_t * valid_agent_mask_t.unsqueeze(-1)
                reappear_error_t = F.smooth_l1_loss(
                    self.reappear_proj(prev_slot_latent_t),
                    enemy_ctx_t.detach(),
                    reduction="none",
                ).mean(dim=-1)
                reappear_sum = reappear_sum + (reappear_error_t * reappear_mask_t).sum()
                reappear_denom = reappear_denom + reappear_mask_t.sum()

            hidden_frac_sum = hidden_frac_sum + (hidden_mask_t.mean(dim=-1) * valid_agent_mask_t).sum()
            hidden_frac_count = hidden_frac_count + valid_agent_mask_t.sum()
            slot_norm_t = belief_slots_t.norm(dim=-1).mean(dim=-1)
            slot_norm_sum = slot_norm_sum + (slot_norm_t * valid_agent_mask_t).sum()
            slot_norm_count = slot_norm_count + valid_agent_mask_t.sum()
            context_norm_t = belief_context_t.norm(dim=-1)
            context_norm_sum = context_norm_sum + (context_norm_t * valid_agent_mask_t).sum()
            context_norm_count = context_norm_count + valid_agent_mask_t.sum()
            belief_attn_clamped_t = belief_attn_t.clamp(min=1e-6)
            attn_entropy_t = -(belief_attn_clamped_t * belief_attn_clamped_t.log()).sum(dim=-1)
            attn_entropy_sum = attn_entropy_sum + (attn_entropy_t * valid_agent_mask_t).sum()
            attn_entropy_count = attn_entropy_count + valid_agent_mask_t.sum()
            top1_mass_t = belief_attn_t.max(dim=-1)[0]
            top1_mass_sum = top1_mass_sum + (top1_mass_t * valid_agent_mask_t).sum()
            top1_mass_count = top1_mass_count + valid_agent_mask_t.sum()

            if not self.use_teacher_targets:
                prev_slot_latent_t = slot_latent_t
                prev_enemy_visible_t = enemy_visible_t
                continue

            ally_state_t = batch["state"][:, t, :self.ally_state_dim]
            ally_state_t = ally_state_t.reshape(batch.batch_size, self.args.n_agents, self.ally_state_feat_dim)
            own_state_t = ally_state_t
            ally_mean_agent_t = ally_state_t.mean(dim=1, keepdim=True).expand(-1, self.args.n_agents, -1)
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
            (
                teacher_attn_t,
                teacher_context_t,
                teacher_full_q_t,
                teacher_chosen_q_t,
                teacher_value_mask_t,
                teacher_delta_t,
                teacher_eval_count_t,
            ) = self._build_value_teacher(
                q_context_t,
                teacher_enemy_feat_t,
                hidden_present_mask_t,
                actions[:, t],
            )
            teacher_value_mask_t = teacher_value_mask_t * valid_agent_mask_t
            hidden_row_mask_t = (hidden_present_mask_t.sum(dim=-1) > 0).float() * valid_agent_mask_t
            teacher_eval_count_sum = teacher_eval_count_sum + (teacher_eval_count_t * hidden_row_mask_t).sum()
            teacher_eval_frac_t = teacher_eval_count_t / hidden_present_mask_t.sum(dim=-1).clamp(min=1.0)
            teacher_eval_frac_sum = teacher_eval_frac_sum + (teacher_eval_frac_t * hidden_row_mask_t).sum()
            teacher_eval_frac_count = teacher_eval_frac_count + hidden_row_mask_t.sum()

            if self.use_latent_align:
                latent_error_t = F.smooth_l1_loss(
                    slot_latent_t,
                    teacher_enemy_feat_t.detach(),
                    reduction="none",
                ).mean(dim=-1)
                latent_mask_t = entity_present_t * valid_agent_mask_t.unsqueeze(-1)
                latent_align_sum = latent_align_sum + (latent_error_t * latent_mask_t).sum()
                latent_align_denom = latent_align_denom + latent_mask_t.sum()

            if self.use_attn_align:
                valid_rows_t = teacher_value_mask_t > 0
                if valid_rows_t.any():
                    teacher_prob_rows = teacher_attn_t[valid_rows_t]
                    student_prob_rows = belief_attn_t[valid_rows_t].clamp(min=1e-6)
                    attn_align_sum = attn_align_sum - (
                        teacher_prob_rows * student_prob_rows.log()
                    ).sum()
                    attn_align_denom = attn_align_denom + teacher_prob_rows.new_tensor(
                        float(valid_rows_t.sum().item())
                    )

            if self.use_context_align:
                context_error_t = F.smooth_l1_loss(
                    belief_context_t,
                    teacher_context_t.detach(),
                    reduction="none",
                ).mean(dim=-1)
                context_align_sum = context_align_sum + (context_error_t * teacher_value_mask_t).sum()
                context_align_denom = context_align_denom + teacher_value_mask_t.sum()

            if self.use_teacher_q:
                teacher_chosen_q_list.append(teacher_chosen_q_t)
                teacher_global_mask_list.append(teacher_value_mask_t)

            teacher_context_norm_t = teacher_context_t.norm(dim=-1)
            teacher_context_norm_sum = teacher_context_norm_sum + (
                teacher_context_norm_t * teacher_value_mask_t
            ).sum()
            teacher_context_norm_count = teacher_context_norm_count + teacher_value_mask_t.sum()
            teacher_attn_clamped_t = teacher_attn_t.clamp(min=1e-6)
            teacher_attn_entropy_t = -(
                teacher_attn_clamped_t * teacher_attn_clamped_t.log()
            ).sum(dim=-1)
            teacher_attn_entropy_sum = teacher_attn_entropy_sum + (
                teacher_attn_entropy_t * teacher_value_mask_t
            ).sum()
            teacher_attn_entropy_count = teacher_attn_entropy_count + teacher_value_mask_t.sum()
            teacher_top1_mass_t = teacher_attn_t.max(dim=-1)[0]
            teacher_top1_mass_sum = teacher_top1_mass_sum + (
                teacher_top1_mass_t * teacher_value_mask_t
            ).sum()
            teacher_top1_mass_count = teacher_top1_mass_count + teacher_value_mask_t.sum()
            teacher_delta_mean_t = teacher_delta_t.sum(dim=-1) / hidden_present_mask_t.sum(dim=-1).clamp(min=1.0)
            teacher_delta_mean_sum = teacher_delta_mean_sum + (
                teacher_delta_mean_t * teacher_value_mask_t
            ).sum()
            teacher_delta_mean_count = teacher_delta_mean_count + teacher_value_mask_t.sum()
            teacher_delta_max_t = teacher_delta_t.max(dim=-1)[0]
            teacher_delta_max_sum = teacher_delta_max_sum + (
                teacher_delta_max_t * teacher_value_mask_t
            ).sum()
            teacher_delta_max_count = teacher_delta_max_count + teacher_value_mask_t.sum()
            prev_slot_latent_t = slot_latent_t
            prev_enemy_visible_t = enemy_visible_t

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

        latent_align_loss = latent_align_sum / latent_align_denom.clamp(min=1.0)
        attn_align_loss = attn_align_sum / attn_align_denom.clamp(min=1.0)
        context_align_loss = context_align_sum / context_align_denom.clamp(min=1.0)
        reappear_loss = reappear_sum / reappear_denom.clamp(min=1.0)
        teacher_q_loss = rewards.new_tensor(0.0)
        teacher_q_weight = self.teacher_q_coef * warm if self.use_teacher_q else 0.0

        if self.use_teacher_q and teacher_chosen_q_list:
            teacher_chosen_action_qvals = th.stack(teacher_chosen_q_list, dim=1)
            if self.mixer is not None:
                teacher_chosen_action_qvals = self.mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])
            teacher_td_error = teacher_chosen_action_qvals - targets.detach()
            teacher_masked_td_error = teacher_td_error * q_mask
            teacher_global_mask = th.stack(teacher_global_mask_list, dim=1)
            if self.mixer is not None:
                teacher_global_mask = (teacher_global_mask.sum(dim=-1, keepdim=True) > 0).float()
            teacher_q_loss = (
                (teacher_masked_td_error ** 2) * teacher_global_mask
            ).sum() / teacher_global_mask.sum().clamp(min=1.0)

        latent_align_weight = self.belief_latent_align_coef * warm
        attn_align_weight = self.belief_attn_align_coef * warm
        context_align_weight = self.belief_context_align_coef * warm
        reappear_weight = self.belief_reappear_coef * warm
        loss = (
            q_loss
            + teacher_q_weight * teacher_q_loss
            + latent_align_weight * latent_align_loss
            + attn_align_weight * attn_align_loss
            + context_align_weight * context_align_loss
            + reappear_weight * reappear_loss
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
            self.logger.log_stat("belief_teacher_q_loss", teacher_q_loss.item(), t_env)
            self.logger.log_stat("belief_teacher_q_weight", teacher_q_weight, t_env)
            self.logger.log_stat("belief_latent_align_loss", latent_align_loss.item(), t_env)
            self.logger.log_stat("belief_latent_align_weight", latent_align_weight, t_env)
            self.logger.log_stat("belief_attn_align_loss", attn_align_loss.item(), t_env)
            self.logger.log_stat("belief_attn_align_weight", attn_align_weight, t_env)
            self.logger.log_stat("belief_context_align_loss", context_align_loss.item(), t_env)
            self.logger.log_stat("belief_context_align_weight", context_align_weight, t_env)
            self.logger.log_stat("belief_reappear_loss", reappear_loss.item(), t_env)
            self.logger.log_stat("belief_reappear_weight", reappear_weight, t_env)
            self.logger.log_stat("belief_reappear_count", reappear_denom.item(), t_env)
            self.logger.log_stat(
                "belief_hidden_frac",
                (hidden_frac_sum / hidden_frac_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_slot_norm",
                (slot_norm_sum / slot_norm_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_context_norm",
                (context_norm_sum / context_norm_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_context_norm",
                (teacher_context_norm_sum / teacher_context_norm_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_attn_entropy",
                (attn_entropy_sum / attn_entropy_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_attn_entropy",
                (teacher_attn_entropy_sum / teacher_attn_entropy_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_top1_mass",
                (top1_mass_sum / top1_mass_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_top1_mass",
                (teacher_top1_mass_sum / teacher_top1_mass_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_delta_mean",
                (teacher_delta_mean_sum / teacher_delta_mean_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_delta_max",
                (teacher_delta_max_sum / teacher_delta_max_count.clamp(min=1.0)).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_value_rows",
                teacher_delta_mean_count.item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_loo_count",
                teacher_eval_count_sum.item() / teacher_eval_frac_count.clamp(min=1.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "belief_teacher_loo_frac",
                (teacher_eval_frac_sum / teacher_eval_frac_count.clamp(min=1.0)).item(),
                t_env,
            )
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
        self.teacher_enemy_encoder.cuda()
        self.teacher_query_proj.cuda()
        self.teacher_key_proj.cuda()
        self.teacher_q_head.cuda()
        self.reappear_proj.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.teacher_enemy_encoder.state_dict(), "{}/teacher_enemy_encoder.th".format(path))
        th.save(self.teacher_query_proj.state_dict(), "{}/teacher_query_proj.th".format(path))
        th.save(self.teacher_key_proj.state_dict(), "{}/teacher_key_proj.th".format(path))
        th.save(self.teacher_q_head.state_dict(), "{}/teacher_q_head.th".format(path))
        th.save(self.reappear_proj.state_dict(), "{}/reappear_proj.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.teacher_enemy_encoder.load_state_dict(
            th.load("{}/teacher_enemy_encoder.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.teacher_query_proj.load_state_dict(
            th.load("{}/teacher_query_proj.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.teacher_key_proj.load_state_dict(
            th.load("{}/teacher_key_proj.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.teacher_q_head.load_state_dict(
            th.load("{}/teacher_q_head.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.reappear_proj.load_state_dict(
            th.load("{}/reappear_proj.th".format(path), map_location=lambda storage, loc: storage)
        )
