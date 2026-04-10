import torch
import torch.nn as nn
import torch.nn.functional as F


class EnemyHistoryTransformerBeliefAgent(nn.Module):
    """Transformer agent with enemy-only history tokens.

    与 history-token 版本保持同样的接口与输出；
    仅在 Transformer 输入前加入“敌人历史压缩”步骤，
    用固定数量 latent token 汇聚可见敌人历史，降低序列长度与显存。
    """

    def __init__(self, input_shape, args):
        super(EnemyHistoryTransformerBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.history_steps = getattr(args, "history_steps", 4)
        self.tokens_per_step = args.history_tokens_per_step
        self.current_context_tokens = getattr(args, "current_context_tokens", 0)
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)
        self.enemy_history_latents = max(1, int(getattr(args, "enemy_history_latents", min(8, self.n_enemies))))

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        # 编码序列长度：hidden token + 当前上下文 token + 固定个数的敌人历史 latent token
        self.max_seq_len = 1 + self.current_context_tokens + self.enemy_history_latents
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_dim))

        # 用可学习查询向量将变长敌人历史压缩为固定长度，减少注意力开销
        self.enemy_history_queries = nn.Parameter(torch.zeros(1, self.enemy_history_latents, self.hidden_dim))
        nn.init.xavier_uniform_(self.enemy_history_queries)
        self.enemy_history_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.enemy_history_norm = nn.LayerNorm(self.hidden_dim)

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
        self.q_head = nn.Linear(self.hidden_dim * 5, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def _masked_mean(self, feats, mask):
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _compress_enemy_history(self, history_emb):
        # history_emb = [B, T, H]，前 current_context_tokens 是当前上下文，后面是敌人历史
        if self.current_context_tokens > 0:
            split = min(self.current_context_tokens, history_emb.size(1))
            context_emb = history_emb[:, :split, :]
            enemy_hist_emb = history_emb[:, split:, :]
        else:
            context_emb = history_emb[:, :0, :]
            enemy_hist_emb = history_emb

        latent_q = self.enemy_history_queries.expand(history_emb.size(0), -1, -1)
        if enemy_hist_emb.size(1) > 0:
            compressed_enemy, _ = self.enemy_history_attn(latent_q, enemy_hist_emb, enemy_hist_emb, need_weights=False)
            compressed_enemy = self.enemy_history_norm(latent_q + compressed_enemy)
        else:
            compressed_enemy = latent_q

        return context_emb, compressed_enemy

    def forward(self, history_tokens, current_step, hidden_state):
        history_emb = F.relu(self.token_embed(history_tokens))
        h_in = hidden_state.reshape(-1, self.hidden_dim).unsqueeze(1)

        context_emb, compressed_enemy = self._compress_enemy_history(history_emb)
        seq = torch.cat([h_in, context_emb, compressed_enemy], dim=1)
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
        visible_enemy_feat = self._masked_mean(
            F.relu(self.token_embed(current_step["enemy_tokens"])),
            current_step["enemy_visible"],
        )
        unseen_enemy_feat = self._masked_mean(
            F.relu(self.belief_enemy_proj(belief_mu)),
            1.0 - current_step["enemy_visible"].float(),
        )

        q_input = torch.cat(
            [memory, move_feat, self_feat, visible_enemy_feat, unseen_enemy_feat],
            dim=-1,
        )
        q = self.q_head(q_input)

        return q, memory, belief_mu, belief_logvar, belief_u
