import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTransformerBeliefAgent(nn.Module):
    """Transformer agent over current-step tokens only.

    The token sequence contains the recurrent memory token plus the
    current step's local observation tokens. This keeps the new
    token_belief path aligned with its name: no explicit history
    window is maintained outside the latent memory.
    """

    def __init__(self, input_shape, args):
        super(TokenTransformerBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.tokens_per_step = args.history_tokens_per_step
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        # token_belief uses only the current step, so the sequence is:
        # [memory_token] + [current_step_tokens]
        self.max_seq_len = 1 + self.tokens_per_step
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

        belief_out_dim = self.n_enemies * self.enemy_state_feat_dim
        self.belief_mu_head = nn.Linear(self.hidden_dim, belief_out_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, belief_out_dim)
        self.belief_u_head = nn.Linear(
            self.hidden_dim,
            belief_out_dim * self.belief_lowrank_rank,
        )

        self.belief_enemy_proj = nn.Linear(self.enemy_state_feat_dim, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def _masked_mean(self, feats, mask):
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def forward(self, step_tokens, current_step, hidden_state):
        step_emb = F.relu(self.token_embed(step_tokens))
        h_in = hidden_state.reshape(-1, self.hidden_dim).unsqueeze(1)

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
