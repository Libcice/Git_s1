import torch
import torch.nn as nn
import torch.nn.functional as F

from .TrXL import build_trxl_encoder


class TrXLBeliefAgent(nn.Module):
    """Belief agent with [memory token] + [current-step tokens] only."""

    def __init__(self, input_shape, args):
        super(TrXLBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.tokens_per_step = getattr(args, "trxl_tokens_per_step")
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)
        self.trxl_impl = getattr(args, "trxl_impl", "builtin")
        self.trxl_activation = getattr(args, "trxl_activation", "relu")
        self.trxl_norm_first = getattr(args, "trxl_norm_first", False)
        self.trxl_final_norm = getattr(args, "trxl_final_norm", False)
        self.trxl_ff_mult = getattr(args, "trxl_ff_mult", 4)

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.n_allies = max(int(args.n_agents) - 1, 0)
        self.max_seq_len = 1 + self.tokens_per_step
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_dim))
        # Add a stable entity identity on top of the raw token content.
        self.memory_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.move_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.self_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.enemy_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.ally_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.enemy_index_embed = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))
        if self.n_allies > 0:
            self.ally_index_embed = nn.Parameter(torch.zeros(1, self.n_allies, self.hidden_dim))
        else:
            self.register_parameter('ally_index_embed', None)
        self.encoder = build_trxl_encoder(
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            activation=self.trxl_activation,
            norm_first=self.trxl_norm_first,
            final_norm=self.trxl_final_norm,
            ff_mult=self.trxl_ff_mult,
            impl=self.trxl_impl,
        )

        belief_out_dim = self.n_enemies * self.enemy_state_feat_dim
        self.belief_mu_head = nn.Linear(self.hidden_dim, belief_out_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, belief_out_dim)
        self.belief_u_head = nn.Linear(
            self.hidden_dim,
            belief_out_dim * self.belief_lowrank_rank,
        )

        # q_head only consumes the unseen-enemy mean belief state.
        self.belief_enemy_proj = nn.Linear(self.enemy_state_feat_dim, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def _masked_mean(self, feats, mask):
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _add_entity_encoding(self, step_emb):
        # Token order is fixed: move, self, enemy_0..N, ally_0..M.
        move_token = step_emb[:, :1, :] + self.move_type
        self_token = step_emb[:, 1:2, :] + self.self_type
        cursor = 2
        enemy_tokens = (
            step_emb[:, cursor:cursor + self.n_enemies, :]
            + self.enemy_type
            + self.enemy_index_embed
        )
        cursor += self.n_enemies
        ally_tokens = step_emb[:, cursor:cursor + self.n_allies, :]
        if self.n_allies > 0:
            ally_tokens = ally_tokens + self.ally_type + self.ally_index_embed
        return torch.cat([move_token, self_token, enemy_tokens, ally_tokens], dim=1)

    def forward(self, step_tokens, current_step, hidden_state):
        step_emb = F.relu(self.token_embed(step_tokens))
        step_emb = self._add_entity_encoding(step_emb)
        h_in = hidden_state.reshape(-1, self.hidden_dim).unsqueeze(1) + self.memory_type

        seq = torch.cat([h_in, step_emb], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1)]
        encoded = self.encoder(seq)

        memory = encoded[:, 0, :]
        belief_mu = self.belief_mu_head(memory).view(
            -1, self.n_enemies, self.enemy_state_feat_dim
        )
        belief_logvar = self.belief_logvar_head(memory).view(
            -1, self.n_enemies, self.enemy_state_feat_dim
        )
        belief_u = self.belief_u_head(memory).view(
            -1,
            self.n_enemies,
            self.enemy_state_feat_dim,
            self.belief_lowrank_rank,
        )

        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        ally_feat = self._masked_mean(
            F.relu(self.token_embed(current_step["ally_tokens"])),
            current_step["ally_visible"],
        )
        visible_enemy_feat = self._masked_mean(
            F.relu(self.token_embed(current_step["enemy_tokens"])),
            current_step["enemy_visible"],
        )
        unseen_enemy_feat = self._masked_mean(
            F.relu(self.belief_enemy_proj(belief_mu)),
            1.0 - current_step["enemy_visible"].float(),
        )

        q_input = torch.cat(
            [memory, move_feat, self_feat, ally_feat, visible_enemy_feat, unseen_enemy_feat],
            dim=-1,
        )
        q = self.q_head(q_input)

        return q, memory, belief_mu, belief_logvar, belief_u
