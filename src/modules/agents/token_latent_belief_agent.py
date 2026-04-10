import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLatentBeliefAgent(nn.Module):
    """Current-step token transformer with global prior/posterior belief latents.

    The transformer hidden state remains the only temporal memory. A global
    prior belief latent is decoded from the previous memory and previous action,
    then updated into a posterior latent using the current-step observation
    context. Per-enemy belief features and value features are decoded from that
    posterior latent for downstream Q estimation.
    """

    def __init__(self, input_shape, args):
        super(TokenLatentBeliefAgent, self).__init__()
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

        self.obs_ctx_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.prior_aux_dim = 2
        self.global_prior_aux_proj = nn.Linear(self.prior_aux_dim, self.hidden_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.prior_ln = nn.LayerNorm(self.hidden_dim)

        self.posterior_delta = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.posterior_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.posterior_ln = nn.LayerNorm(self.hidden_dim)

        self.enemy_prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.enemy_prior_ln = nn.LayerNorm(self.hidden_dim)

        self.enemy_post_delta = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 1 + self.prior_aux_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.enemy_post_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + 1 + self.prior_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.enemy_post_ln = nn.LayerNorm(self.hidden_dim)

        self.belief_mu_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.belief_u_head = nn.Linear(
            self.hidden_dim,
            self.enemy_state_feat_dim * self.belief_lowrank_rank,
        )

        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_importance = nn.Linear(self.hidden_dim, 1)
        self.q_gate = nn.Linear(self.hidden_dim, self.hidden_dim * 5)
        self.q_bias = nn.Linear(self.hidden_dim, self.hidden_dim * 5)
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

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
        return current_memory, move_feat, self_feat, enemy_ctx, ally_ctx

    def _belief_stats_from_feats(self, enemy_feats):
        belief_mu = self.belief_mu_head(enemy_feats)
        belief_logvar = self.belief_logvar_head(enemy_feats)
        belief_u = self.belief_u_head(enemy_feats).view(
            -1,
            self.n_enemies,
            self.enemy_state_feat_dim,
            self.belief_lowrank_rank,
        )
        return belief_mu, belief_logvar, belief_u

    def forward(self, current_step, prev_memory):
        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))
        current_memory, move_feat, self_feat, enemy_ctx, ally_ctx = self._encode_current_step(current_step, prev_memory)
        prior_aux = self._build_prior_aux(current_step)

        ally_summary = self._masked_mean(ally_ctx, current_step["ally_visible"])
        visible_enemy_summary = self._masked_mean(enemy_ctx, current_step["enemy_visible"])
        obs_ctx = F.relu(
            self.obs_ctx_proj(
                torch.cat([current_memory, move_feat, self_feat, ally_summary, visible_enemy_summary], dim=-1)
            )
        )

        prior_aux_summary = F.relu(self.global_prior_aux_proj(prior_aux.mean(dim=1)))
        prior_latent = self.prior_ln(
            self.prior_mlp(torch.cat([prev_memory, prev_action_feat, prior_aux_summary], dim=-1))
        )
        posterior_input = torch.cat([prior_latent, obs_ctx], dim=-1)
        posterior_latent = self.posterior_ln(
            prior_latent
            + self.posterior_gate(posterior_input) * self.posterior_delta(posterior_input)
        )

        prior_enemy_input = torch.cat(
            [
                prior_latent.unsqueeze(1).expand(-1, self.n_enemies, -1),
                self.enemy_index_embed.expand(prev_memory.size(0), -1, -1),
                prior_aux,
            ],
            dim=-1,
        )
        prior_enemy_feats = self.enemy_prior_ln(self.enemy_prior_mlp(prior_enemy_input))

        posterior_enemy_input = torch.cat(
            [
                prior_enemy_feats,
                enemy_ctx,
                posterior_latent.unsqueeze(1).expand(-1, self.n_enemies, -1),
                current_step["enemy_visible"].float().unsqueeze(-1),
                prior_aux,
            ],
            dim=-1,
        )
        corrected_enemy_feats = self.enemy_post_ln(
            prior_enemy_feats
            + self.enemy_post_gate(posterior_enemy_input) * self.enemy_post_delta(posterior_enemy_input)
        )
        visible_mask = current_step["enemy_visible"].float().unsqueeze(-1)
        belief_enemy_feats = visible_mask * corrected_enemy_feats + (1.0 - visible_mask) * prior_enemy_feats

        prior_mu, prior_logvar, prior_u = self._belief_stats_from_feats(prior_enemy_feats)
        belief_mu, belief_logvar, belief_u = self._belief_stats_from_feats(belief_enemy_feats)

        belief_value_feats = F.relu(self.value_proj(belief_enemy_feats))
        if self.disable_belief_for_q:
            # Strict no-belief control: Q only uses transformer memory and
            # directly observed context, without posterior-latent modulation.
            hidden_enemy_summary = torch.zeros_like(visible_enemy_summary)
            q_prefix = torch.cat(
                [current_memory, move_feat, self_feat, ally_summary, visible_enemy_summary],
                dim=-1,
            )
        else:
            confidence = torch.sigmoid(-belief_logvar.mean(dim=-1))
            importance = torch.sigmoid(self.value_importance(belief_value_feats).squeeze(-1))
            hidden_weight = (1.0 - current_step["enemy_visible"].float()) * confidence * importance
            hidden_enemy_summary = self.belief_q_alpha * self._masked_mean(belief_value_feats, hidden_weight)
            q_prefix_base = torch.cat(
                [current_memory, move_feat, self_feat, ally_summary, visible_enemy_summary],
                dim=-1,
            )
            q_prefix = q_prefix_base * torch.sigmoid(self.q_gate(posterior_latent)) + self.q_bias(posterior_latent)
        q_input = torch.cat([q_prefix, hidden_enemy_summary], dim=-1)
        q = self.q_head(q_input)

        return (
            q,
            current_memory,
            prior_mu,
            prior_logvar,
            prior_u,
            belief_mu,
            belief_logvar,
            belief_u,
            q_prefix,
            belief_value_feats,
        )
