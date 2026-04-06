import torch
import torch.nn as nn
import torch.nn.functional as F


class CurrentSlotBeliefRNNAgent(nn.Module):
    """Current-step slot-belief agent without transformer."""

    def __init__(self, input_shape, args):
        super(CurrentSlotBeliefRNNAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "rnn_hidden_dim", 64)
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = max(1, getattr(args, "belief_lowrank_rank", 1))
        self.disable_belief_for_q = getattr(args, "belief_loss_coef", 0.001) <= 0.0
        self.belief_q_alpha = 1.0

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.prev_action_embed = nn.Linear(args.n_actions, self.hidden_dim)
        self.init_belief_slots = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))

        self.prior_aux_dim = 2
        self.prior_gru = nn.GRUCell(
            input_size=self.hidden_dim * 2 + self.prior_aux_dim,
            hidden_size=self.hidden_dim,
        )

        self.memory_fc = nn.Linear(self.hidden_dim * 5, self.hidden_dim)
        self.memory_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.posterior_delta = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.posterior_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.posterior_ln = nn.LayerNorm(self.hidden_dim)

        self.belief_mu_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)

        self.q_slot_aux_dim = 2
        self.slot_to_q = nn.Linear(self.hidden_dim + self.q_slot_aux_dim, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def init_belief(self):
        return self.init_belief_slots

    def set_belief_q_alpha(self, alpha):
        self.belief_q_alpha = float(alpha)

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _build_prior_aux(self, current_step):
        prev_visible = current_step["prev_enemy_visible"].float().unsqueeze(-1)
        unseen_feat = torch.tanh(current_step["unseen_steps"].float().unsqueeze(-1) / 5.0)
        return torch.cat([prev_visible, unseen_feat], dim=-1)

    def _build_q_slot_aux(self, current_step):
        visible = current_step["enemy_visible"].float().unsqueeze(-1)
        unseen_feat = torch.tanh(current_step["unseen_steps"].float().unsqueeze(-1) / 5.0)
        return torch.cat([visible, unseen_feat], dim=-1)

    def _belief_stats_from_slots(self, slots):
        belief_mu = self.belief_mu_head(slots)
        belief_logvar = self.belief_logvar_head(slots)
        belief_u = slots.new_zeros(
            slots.size(0),
            self.n_enemies,
            self.enemy_state_feat_dim,
            self.belief_lowrank_rank,
        )
        return belief_mu, belief_logvar, belief_u

    def forward(self, current_step, prev_belief_slots, prev_memory):
        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        enemy_feat = F.relu(self.token_embed(current_step["enemy_tokens"]))
        ally_feat = F.relu(self.token_embed(current_step["ally_tokens"]))
        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))

        memory_expand = prev_memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        action_expand = prev_action_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        prior_aux = self._build_prior_aux(current_step)
        prior_input = torch.cat([memory_expand, action_expand, prior_aux], dim=-1)
        prior_slots = self.prior_gru(
            prior_input.reshape(-1, prior_input.size(-1)),
            prev_belief_slots.reshape(-1, self.hidden_dim),
        ).view(-1, self.n_enemies, self.hidden_dim)

        ally_summary = self._masked_mean(ally_feat, current_step["ally_visible"])
        visible_enemy_summary = self._masked_mean(enemy_feat, current_step["enemy_visible"])
        prior_slot_summary = prior_slots.mean(dim=1)

        memory_input = torch.cat(
            [move_feat, self_feat, ally_summary, visible_enemy_summary, prior_slot_summary],
            dim=-1,
        )
        memory_x = F.relu(self.memory_fc(memory_input))
        current_memory = self.memory_gru(memory_x, prev_memory)

        corr_input = torch.cat(
            [
                prior_slots,
                enemy_feat,
                current_memory.unsqueeze(1).expand(-1, self.n_enemies, -1),
            ],
            dim=-1,
        )
        corrected_slots = self.posterior_ln(
            prior_slots + self.posterior_gate(corr_input) * self.posterior_delta(corr_input)
        )
        visible_mask = current_step["enemy_visible"].float().unsqueeze(-1)
        post_slots = visible_mask * corrected_slots + (1.0 - visible_mask) * prior_slots

        prior_mu, prior_logvar, prior_u = self._belief_stats_from_slots(prior_slots)
        belief_mu, belief_logvar, belief_u = self._belief_stats_from_slots(post_slots)

        if self.disable_belief_for_q:
            hidden_enemy_summary = torch.zeros_like(visible_enemy_summary)
        else:
            q_slot_aux = self._build_q_slot_aux(current_step)
            slot_q_feats = F.relu(self.slot_to_q(torch.cat([post_slots, q_slot_aux], dim=-1)))
            confidence = torch.sigmoid(-belief_logvar.mean(dim=-1))
            hidden_weight = (1.0 - current_step["enemy_visible"].float()) * confidence
            hidden_enemy_summary = self.belief_q_alpha * self._masked_mean(slot_q_feats, hidden_weight)

        q_input = torch.cat(
            [
                current_memory,
                move_feat,
                self_feat,
                ally_summary,
                visible_enemy_summary,
                hidden_enemy_summary,
            ],
            dim=-1,
        )
        q = self.q_head(q_input)

        return (
            q,
            current_memory,
            post_slots,
            prior_mu,
            prior_logvar,
            prior_u,
            belief_mu,
            belief_logvar,
            belief_u,
        )
