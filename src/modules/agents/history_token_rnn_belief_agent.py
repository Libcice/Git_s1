import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryTokenRNNBeliefAgent(nn.Module):
    """RNN agent over a short token history.

    Tokens are built only from local observations. This mirrors the
    history-token transformer belief path, but uses a recurrent encoder
    for the history sequence.
    """

    def __init__(self, input_shape, args):
        super(HistoryTokenRNNBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "rnn_hidden_dim", 64)
        self.history_steps = getattr(args, "history_steps", 4)
        self.tokens_per_step = args.history_tokens_per_step
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.history_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

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

    def _encode_history(self, history_tokens, hidden_state):
        h0 = hidden_state.reshape(1, -1, self.hidden_dim).contiguous()
        if history_tokens.size(1) == 0:
            return h0.squeeze(0)

        history_emb = F.relu(self.token_embed(history_tokens))
        output, hn = self.history_rnn(history_emb, h0)
        return hn.squeeze(0)

    def forward(self, history_tokens, current_step, hidden_state):
        memory = self._encode_history(history_tokens, hidden_state)

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
