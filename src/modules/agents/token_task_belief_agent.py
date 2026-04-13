import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTaskBeliefAgent(nn.Module):
    """Token-transformer agent with query-conditioned belief readout.

    The recurrent memory summarises visible history and current task context.
    Per-enemy belief slots store object-centric hidden state. A query produced
    from the visible context reads the hidden slots through attention, and a
    single Q head consumes the resulting belief context.
    """

    def __init__(self, input_shape, args):
        super(TokenTaskBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.n_enemies = args.enemy_num
        self.n_allies = args.n_agents - 1
        self.belief_topk = max(0, int(getattr(args, "belief_topk", 2)))
        self.disable_belief = (
            getattr(args, "belief_latent_align_coef", 0.10) <= 0.0
            and getattr(args, "belief_attn_align_coef", 0.05) <= 0.0
            and getattr(args, "belief_context_align_coef", 0.10) <= 0.0
            and getattr(args, "belief_teacher_q_coef", 0.10) <= 0.0
            and getattr(args, "belief_reappear_coef", 0.10) <= 0.0
        )

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
        self.slot_latent_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.slot_latent_ln = nn.LayerNorm(self.hidden_dim)
        self.slot_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.slot_value_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)
        self.attn_scale = 1.0 / math.sqrt(float(self.hidden_dim))

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def init_belief_slots(self):
        return self.token_embed.weight.new_zeros(1, self.n_enemies, self.hidden_dim)

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

    def _masked_topk_softmax(self, logits, mask):
        mask_bool = mask > 0
        masked_logits = logits.masked_fill(~mask_bool, -1e9)

        if 0 < self.belief_topk < logits.size(-1):
            topk_k = min(self.belief_topk, logits.size(-1))
            topk_idx = masked_logits.topk(k=topk_k, dim=-1).indices
            topk_mask = torch.zeros_like(mask_bool)
            topk_mask.scatter_(1, topk_idx, True)
            mask_bool = mask_bool & topk_mask
            masked_logits = logits.masked_fill(~mask_bool, -1e9)

        probs = F.softmax(masked_logits, dim=-1)
        probs = probs * mask_bool.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        empty_rows = mask_bool.float().sum(dim=-1, keepdim=True) <= 0
        probs = torch.where(empty_rows, torch.zeros_like(probs), probs)
        return probs

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
        q_context = torch.cat(
            [current_memory, move_feat, self_feat, raw_ally_summary, raw_visible_enemy_summary],
            dim=-1,
        )
        query = self.query_proj(q_context)

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

        slot_latent = self.slot_latent_ln(
            new_belief_slots + self.slot_latent_proj(torch.cat([new_belief_slots, prior_aux], dim=-1))
        )

        hidden_mask = 1.0 - current_step["enemy_visible"].float()
        if self.disable_belief:
            belief_attn = torch.zeros_like(hidden_mask)
            belief_context = q_context.new_zeros(q_context.size(0), self.hidden_dim)
        else:
            slot_keys = self.slot_key_proj(slot_latent)
            attn_logits = (query.unsqueeze(1) * slot_keys).sum(dim=-1) * self.attn_scale
            belief_attn = self._masked_topk_softmax(attn_logits, hidden_mask)
            belief_context = (
                belief_attn.unsqueeze(-1) * self.slot_value_proj(slot_latent)
            ).sum(dim=1)

        q = self.q_head(torch.cat([q_context, belief_context], dim=-1))

        return (
            q,
            current_memory,
            new_belief_slots,
            slot_latent,
            belief_attn,
            belief_context,
            q_context,
            enemy_ctx,
        )
