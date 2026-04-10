import torch
import torch.nn as nn
import torch.nn.functional as F


class HistorySlotBeliefAgent(nn.Module):
    """Current-step transformer with persistent per-enemy belief slots."""

    def __init__(self, input_shape, args):
        super(HistorySlotBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.n_enemies = args.enemy_num
        self.n_allies = args.n_agents - 1
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = max(1, getattr(args, "belief_lowrank_rank", 1))
        self.disable_belief_for_q = getattr(args, "belief_loss_coef", 0.001) <= 0.0
        self.belief_q_alpha = 1.0

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.prev_action_embed = nn.Linear(args.n_actions, self.hidden_dim)
        self.init_belief_slots = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))

        self.memory_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.move_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.self_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.enemy_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.ally_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.enemy_index_embed = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))
        self.ally_index_embed = nn.Parameter(torch.zeros(1, max(1, self.n_allies), self.hidden_dim))

        self.max_seq_len = 3 + self.n_enemies + self.n_allies
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)

        self.prior_aux_dim = 2
        self.prior_gru = nn.GRUCell(
            input_size=self.hidden_dim * 2 + self.prior_aux_dim,
            hidden_size=self.hidden_dim,
        )

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

    def _encode_current_step(self, current_step, prev_memory):
        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        enemy_feat = F.relu(self.token_embed(current_step["enemy_tokens"]))
        ally_feat = F.relu(self.token_embed(current_step["ally_tokens"]))

        memory_token = prev_memory.unsqueeze(1) + self.memory_type
        move_token = move_feat.unsqueeze(1) + self.move_type
        self_token = self_feat.unsqueeze(1) + self.self_type
        enemy_tokens = enemy_feat + self.enemy_type + self.enemy_index_embed
        ally_tokens = ally_feat + self.ally_type + self.ally_index_embed[:, : ally_feat.size(1)]

        seq = torch.cat([memory_token, move_token, self_token, enemy_tokens, ally_tokens], dim=1)
        seq = seq + self.pos_embed[:, : seq.size(1)]

        bs_agents = seq.size(0)
        fixed_valid = seq.new_ones(bs_agents, 3)
        valid = torch.cat([fixed_valid, current_step["enemy_visible"], current_step["ally_visible"]], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=valid <= 0)

        enemy_start = 3
        enemy_end = enemy_start + self.n_enemies
        current_memory = encoded[:, 0, :]
        enemy_ctx = encoded[:, enemy_start:enemy_end, :]
        ally_ctx = encoded[:, enemy_end:, :]
        return current_memory, move_feat, self_feat, enemy_ctx, ally_ctx

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
        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))
        memory_expand = prev_memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        action_expand = prev_action_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        prior_aux = self._build_prior_aux(current_step)
        prior_input = torch.cat([memory_expand, action_expand, prior_aux], dim=-1)

        prior_slots = self.prior_gru(
            prior_input.reshape(-1, prior_input.size(-1)),
            prev_belief_slots.reshape(-1, self.hidden_dim),
        ).view(-1, self.n_enemies, self.hidden_dim)

        current_memory, move_feat, self_feat, enemy_ctx, ally_ctx = self._encode_current_step(
            current_step,
            prev_memory,
        )

        corr_input = torch.cat(
            [
                prior_slots,
                enemy_ctx,
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

        ally_summary = self._masked_mean(ally_ctx, current_step["ally_visible"])
        visible_enemy_summary = self._masked_mean(enemy_ctx, current_step["enemy_visible"])

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
