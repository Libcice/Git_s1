import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenValueBeliefAgent(nn.Module):
    """Token transformer agent with per-enemy local value-belief slots.

    The transformer still encodes the current-step token layout, while a
    separate slot is maintained for each enemy. Slots are propagated with
    residual-gated prior updates and corrected by visible enemy tokens.
    Each slot predicts how the hidden enemy should modify local Q-values
    rather than reconstructing the full hidden state.
    """

    def __init__(self, input_shape, args):
        super(TokenValueBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.n_enemies = args.enemy_num
        self.n_allies = args.n_agents - 1
        self.value_belief_tau = max(1e-3, float(getattr(args, "belief_decay_tau", 6.0)))
        self.disable_belief_for_q = (
            getattr(args, "belief_delta_total_coef", 0.25) <= 0.0
            and getattr(args, "belief_delta_slot_coef", 0.10) <= 0.0
            and getattr(args, "belief_rank_coef", 0.05) <= 0.0
            and getattr(args, "belief_teacher_q_coef", 0.10) <= 0.0
        )
        self.belief_q_alpha = 1.0

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.prev_action_embed = nn.Linear(args.n_actions, self.hidden_dim)

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
        self.slot_prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.slot_prior_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + self.prior_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.slot_prior_ln = nn.LayerNorm(self.hidden_dim)

        self.slot_post_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 5 + 1 + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.slot_post_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 5 + 1 + self.prior_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.slot_post_ln = nn.LayerNorm(self.hidden_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.delta_q_enemy_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, args.n_actions),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.prior_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.alive_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.hp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.q_visible_head = nn.Linear(self.hidden_dim * 5, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def init_belief_slots(self):
        return self.token_embed.weight.new_zeros(1, self.n_enemies, self.hidden_dim)

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

    def _encode_current_step(self, current_step, prev_memory):
        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        enemy_feat = F.relu(self.token_embed(current_step["enemy_tokens"]))
        ally_feat = F.relu(self.token_embed(current_step["ally_tokens"]))

        memory_token = prev_memory.unsqueeze(1) + self.memory_type
        move_token = move_feat.unsqueeze(1) + self.move_type
        self_token = self_feat.unsqueeze(1) + self.self_type
        enemy_tokens = enemy_feat + self.enemy_type + self.enemy_index_embed
        ally_tokens = ally_feat + self.ally_type + self.ally_index_embed[:, :ally_feat.size(1)]

        seq = torch.cat([memory_token, move_token, self_token, enemy_tokens, ally_tokens], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1)]

        bs_agents = seq.size(0)
        fixed_valid = seq.new_ones(bs_agents, 3)
        valid = torch.cat([fixed_valid, current_step["enemy_visible"], current_step["ally_visible"]], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=valid <= 0)

        enemy_start = 3
        enemy_end = enemy_start + self.n_enemies
        current_memory = encoded[:, 0, :]
        enemy_ctx = encoded[:, enemy_start:enemy_end, :]
        ally_ctx = encoded[:, enemy_end:, :]
        return current_memory, move_feat, self_feat, enemy_feat, ally_feat, enemy_ctx, ally_ctx

    def forward(self, current_step, prev_memory, prev_belief_slots):
        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))
        (
            current_memory,
            move_feat,
            self_feat,
            enemy_feat,
            ally_feat,
            enemy_ctx,
            ally_ctx,
        ) = self._encode_current_step(current_step, prev_memory)
        prior_aux = self._build_prior_aux(current_step)
        visible_mask = current_step["enemy_visible"].float().unsqueeze(-1)

        raw_ally_summary = self._masked_mean(ally_feat, current_step["ally_visible"])
        raw_visible_enemy_summary = self._masked_mean(enemy_feat, current_step["enemy_visible"])
        q_visible_prefix = torch.cat(
            [current_memory, move_feat, self_feat, raw_ally_summary, raw_visible_enemy_summary],
            dim=-1,
        )
        q_visible = self.q_visible_head(q_visible_prefix)
        query = F.relu(self.query_proj(q_visible_prefix))

        prev_memory_expand = prev_memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        prev_action_expand = prev_action_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        enemy_index_expand = self.enemy_index_embed.expand(prev_memory.size(0), -1, -1)
        prior_input = torch.cat(
            [prev_belief_slots, prev_memory_expand, prev_action_expand, enemy_index_expand, prior_aux],
            dim=-1,
        )
        slot_prior = self.slot_prior_ln(
            prev_belief_slots
            + self.slot_prior_gate(prior_input) * self.slot_prior_mlp(prior_input)
        )

        current_memory_expand = current_memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        move_expand = move_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        self_expand = self_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        post_input = torch.cat(
            [
                slot_prior,
                enemy_ctx,
                current_memory_expand,
                move_expand,
                self_expand,
                current_step["enemy_visible"].float().unsqueeze(-1),
                prior_aux,
            ],
            dim=-1,
        )
        slot_post = self.slot_post_ln(
            slot_prior + visible_mask * self.slot_post_gate(post_input) * self.slot_post_mlp(post_input)
        )
        new_belief_slots = visible_mask * slot_post + (1.0 - visible_mask) * slot_prior

        query_expand = query.unsqueeze(1).expand(-1, self.n_enemies, -1)
        delta_input = torch.cat([new_belief_slots, query_expand, prior_aux], dim=-1)
        delta_q_per_enemy = self.delta_q_enemy_head(delta_input)
        rel_logits = self.rel_head(delta_input).squeeze(-1)
        alive_logits = self.alive_head(new_belief_slots).squeeze(-1)
        hp_preds = self.hp_head(new_belief_slots).squeeze(-1)

        if self.disable_belief_for_q:
            enemy_weight = torch.zeros_like(current_step["enemy_visible"].float())
            delta_q_hidden = torch.zeros_like(q_visible)
        else:
            hidden_mask = 1.0 - current_step["enemy_visible"].float()
            age_decay = torch.exp(
                -current_step["unseen_steps"].float().clamp(min=0.0) / self.value_belief_tau
            )
            rel_weight = torch.sigmoid(rel_logits)
            alive_weight = torch.sigmoid(alive_logits)
            enemy_weight = hidden_mask * age_decay * rel_weight * alive_weight
            delta_q_hidden = self.belief_q_alpha * (
                enemy_weight.unsqueeze(-1) * delta_q_per_enemy
            ).sum(dim=1)

        q = q_visible + delta_q_hidden

        return (
            q,
            current_memory,
            new_belief_slots,
            delta_q_per_enemy,
            rel_logits,
            alive_logits,
            hp_preds,
            q_visible,
            delta_q_hidden,
            enemy_weight,
        )
