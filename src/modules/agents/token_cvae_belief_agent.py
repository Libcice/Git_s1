import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim, hidden_dim, out_dim, n_layers=2):
    layers = []
    last_dim = in_dim
    for _ in range(max(1, n_layers - 1)):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class TokenCVAEBeliefAgent(nn.Module):
    """Token-transformer history encoder with a CVAE belief model.

    The transformer hidden state is the only temporal memory. A CVAE is used
    to infer hidden enemy state from history and privileged posterior targets
    during training. The Q path always consumes prior-side belief only.
    """

    def __init__(self, input_shape, args):
        super(TokenCVAEBeliefAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.n_enemies = args.enemy_num
        self.n_allies = args.n_agents - 1
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.latent_dim = getattr(args, "belief_latent_dim", 16)
        self.disable_belief = getattr(args, "disable_belief", False)
        self.use_belief_for_q = getattr(args, "use_belief_for_q", True) and not self.disable_belief
        self.belief_prior_type = getattr(args, "belief_prior_type", "conditional")
        self.belief_logvar_min = getattr(args, "belief_logvar_min", -2.0)
        self.belief_logvar_max = getattr(args, "belief_logvar_max", 1.0)
        default_conf_temp = 0.25 * max(1e-6, self.belief_logvar_max - self.belief_logvar_min)
        self.belief_confidence_center = getattr(
            args,
            "belief_confidence_center",
            0.5 * (self.belief_logvar_min + self.belief_logvar_max),
        )
        self.belief_confidence_temp = max(
            1e-6,
            float(getattr(args, "belief_confidence_temp", default_conf_temp)),
        )

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")
        if self.belief_prior_type not in ("conditional", "standard_normal"):
            raise ValueError("belief_prior_type must be 'conditional' or 'standard_normal'")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.prev_action_embed = nn.Linear(args.n_actions, self.hidden_dim)

        self.memory_type = nn.Parameter(th.zeros(1, 1, self.hidden_dim))
        self.move_type = nn.Parameter(th.zeros(1, 1, self.hidden_dim))
        self.self_type = nn.Parameter(th.zeros(1, 1, self.hidden_dim))
        self.enemy_type = nn.Parameter(th.zeros(1, 1, self.hidden_dim))
        self.ally_type = nn.Parameter(th.zeros(1, 1, self.hidden_dim))
        self.enemy_index_embed = nn.Parameter(th.zeros(1, self.n_enemies, self.hidden_dim))
        self.ally_index_embed = nn.Parameter(th.zeros(1, max(1, self.n_allies), self.hidden_dim))

        self.max_seq_len = 3 + self.n_enemies + self.n_allies
        self.pos_embed = nn.Parameter(th.zeros(1, self.max_seq_len, self.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)

        self.history_context = _build_mlp(self.hidden_dim * 2, self.hidden_dim * 2, self.hidden_dim)
        self.q_context = _build_mlp(self.hidden_dim * 5, self.hidden_dim * 2, self.hidden_dim)

        self.real_enemy_proj = _build_mlp(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.belief_enemy_proj = _build_mlp(self.enemy_state_feat_dim, self.hidden_dim, self.hidden_dim)
        self.hidden_feature_gate_proj = _build_mlp(6, self.hidden_dim, 1)
        self.slot_fusion_proj = _build_mlp(self.hidden_dim * 2 + 6, self.hidden_dim * 2, self.hidden_dim)
        self.enemy_summary_proj = _build_mlp(self.n_enemies * self.hidden_dim, self.hidden_dim * 2, self.hidden_dim)
        self.q_head = _build_mlp(self.hidden_dim * 2, self.hidden_dim * 2, args.n_actions)

        if self.belief_prior_type == "conditional":
            self.prior_encoder = _build_mlp(self.hidden_dim, self.hidden_dim * 2, self.latent_dim * 2)
        else:
            self.prior_encoder = None

        self.posterior_encoder = _build_mlp(
            self.hidden_dim + self.n_enemies * self.enemy_state_feat_dim,
            self.hidden_dim * 2,
            self.latent_dim * 2,
        )
        self.decoder_trunk = _build_mlp(self.hidden_dim + self.latent_dim, self.hidden_dim * 2, self.hidden_dim * 2)
        self.decoder_mu = nn.Linear(self.hidden_dim * 2, self.n_enemies * self.enemy_state_feat_dim)
        self.decoder_logvar = nn.Linear(self.hidden_dim * 2, self.n_enemies * self.enemy_state_feat_dim)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _reparameterize(self, mu, logvar):
        # Keep a defensive clamp here even though all learned logvars are bounded upstream.
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def _bound_logvar(self, raw_logvar):
        span = max(1e-6, self.belief_logvar_max - self.belief_logvar_min)
        return self.belief_logvar_min + span * th.sigmoid(raw_logvar)

    def _split_bounded_stats(self, stats):
        mu, raw_logvar = th.chunk(stats, 2, dim=-1)
        return mu, self._bound_logvar(raw_logvar)

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

        seq = th.cat([memory_token, move_token, self_token, enemy_tokens, ally_tokens], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1)]

        bs_agents = seq.size(0)
        fixed_valid = seq.new_ones(bs_agents, 3)
        valid = th.cat([fixed_valid, current_step["enemy_visible"], current_step["ally_visible"]], dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=valid <= 0)

        enemy_start = 3
        enemy_end = enemy_start + self.n_enemies
        current_memory = encoded[:, 0, :]
        enemy_ctx = encoded[:, enemy_start:enemy_end, :]
        ally_ctx = encoded[:, enemy_end:, :]
        return current_memory, move_feat, self_feat, enemy_feat, ally_feat, enemy_ctx, ally_ctx

    def _history_context(self, current_memory, prev_action_feat):
        return self.history_context(th.cat([current_memory, prev_action_feat], dim=-1))

    def _prior_latent(self, history_context):
        if self.belief_prior_type == "standard_normal":
            prior_mu = history_context.new_zeros(history_context.size(0), self.latent_dim)
            prior_logvar = history_context.new_zeros(history_context.size(0), self.latent_dim)
        else:
            prior_stats = self.prior_encoder(history_context)
            prior_mu, prior_logvar = self._split_bounded_stats(prior_stats)
        return prior_mu, prior_logvar

    def _decode_belief(self, history_context, latent_z):
        decoder_hidden = self.decoder_trunk(th.cat([history_context, latent_z], dim=-1))
        belief_mu = self.decoder_mu(decoder_hidden).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        # Bound uncertainty at the source so downstream code never sees out-of-range logvars.
        belief_logvar = self._bound_logvar(
            self.decoder_logvar(decoder_hidden).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        )
        return belief_mu, belief_logvar

    def _belief_confidence(self, belief_logvar):
        avg_logvar = belief_logvar.clamp(
            min=self.belief_logvar_min,
            max=self.belief_logvar_max,
        ).mean(dim=-1, keepdim=True)
        return th.sigmoid(
            (self.belief_confidence_center - avg_logvar) / self.belief_confidence_temp
        )

    def _posterior_latent(self, history_context, hidden_enemy_state):
        hidden_flat = hidden_enemy_state.reshape(hidden_enemy_state.size(0), -1)
        posterior_stats = self.posterior_encoder(th.cat([history_context, hidden_flat], dim=-1))
        posterior_mu, posterior_logvar = self._split_bounded_stats(posterior_stats)
        return posterior_mu, posterior_logvar

    def forward(self, current_step, prev_memory, hidden_enemy_state=None):
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

        ally_summary = self._masked_mean(ally_feat, current_step["ally_visible"])
        raw_ally_summary = ally_summary
        real_enemy_feat = self.real_enemy_proj(enemy_feat)
        q_context_raw = th.cat(
            [current_memory, move_feat, self_feat, raw_ally_summary, prev_action_feat],
            dim=-1,
        )
        q_context = self.q_context(q_context_raw)

        history_context = self._history_context(current_memory, prev_action_feat)
        prior_z_mu, prior_z_logvar = self._prior_latent(history_context)
        prior_z = prior_z_mu
        prior_belief_mu, prior_belief_logvar = self._decode_belief(history_context, prior_z)

        if self.belief_prior_type == "standard_normal":
            prior_belief_conf = prior_belief_mu.new_ones(prior_belief_mu.size(0), self.n_enemies, 1)
        else:
            prior_belief_conf = self._belief_confidence(prior_belief_logvar)
        prior_belief_feat = F.relu(self.belief_enemy_proj(prior_belief_mu)) * prior_belief_conf

        visible_mask = current_step["enemy_visible"].float().unsqueeze(-1)
        hidden_mask = 1.0 - visible_mask
        visible_frac = current_step["enemy_visible"].float().mean(dim=1, keepdim=True)
        hidden_frac = hidden_mask.squeeze(-1).mean(dim=1, keepdim=True)
        hidden_prior_conf_summary = self._masked_mean(prior_belief_conf, hidden_mask.squeeze(-1))
        belief_conf_slots = prior_belief_conf

        visible_enemy_slots = visible_mask * real_enemy_feat
        hidden_frac_broadcast = hidden_frac.unsqueeze(1).expand(-1, self.n_enemies, -1)
        visible_frac_broadcast = visible_frac.unsqueeze(1).expand(-1, self.n_enemies, -1)
        hidden_prior_conf_summary_broadcast = hidden_prior_conf_summary.unsqueeze(1).expand(-1, self.n_enemies, -1)

        hidden_feature_gate_input = th.cat(
            [
                prior_belief_conf,
                hidden_prior_conf_summary_broadcast,
                visible_mask,
                hidden_mask,
                visible_frac_broadcast,
                hidden_frac_broadcast,
            ],
            dim=-1,
        )
        if self.use_belief_for_q:
            hidden_feature_gate = hidden_mask * th.sigmoid(self.hidden_feature_gate_proj(hidden_feature_gate_input))
            hidden_feature_used = hidden_feature_gate * prior_belief_feat
        else:
            hidden_feature_gate = hidden_mask.new_zeros(hidden_mask.shape)
            hidden_feature_used = prior_belief_feat.new_zeros(prior_belief_feat.shape)
            belief_conf_slots = prior_belief_conf.new_zeros(prior_belief_conf.shape)
            hidden_prior_conf_summary = hidden_prior_conf_summary.new_zeros(hidden_prior_conf_summary.shape)
            hidden_prior_conf_summary_broadcast = hidden_prior_conf_summary_broadcast.new_zeros(
                hidden_prior_conf_summary_broadcast.shape
            )

        slot_fusion_input = th.cat(
            [
                visible_enemy_slots,
                hidden_feature_used,
                visible_mask,
                hidden_mask,
                belief_conf_slots,
                hidden_prior_conf_summary_broadcast,
                visible_frac_broadcast,
                hidden_frac_broadcast,
            ],
            dim=-1,
        )
        fused_enemy_feat = self.slot_fusion_proj(slot_fusion_input)
        # Preserve enemy-slot order until the final compression step so Q can still distinguish slots.
        enemy_summary = self.enemy_summary_proj(fused_enemy_feat.reshape(fused_enemy_feat.size(0), -1))
        q = self.q_head(th.cat([q_context, enemy_summary], dim=-1))

        posterior_belief_mu = None
        posterior_belief_logvar = None
        posterior_z_mu = None
        posterior_z_logvar = None
        if hidden_enemy_state is not None:
            posterior_z_mu, posterior_z_logvar = self._posterior_latent(history_context, hidden_enemy_state)
            posterior_z = self._reparameterize(posterior_z_mu, posterior_z_logvar)
            posterior_belief_mu, posterior_belief_logvar = self._decode_belief(history_context, posterior_z)

        return (
            q,
            current_memory,
            prior_belief_mu,
            prior_belief_logvar,
            posterior_belief_mu,
            posterior_belief_logvar,
            prior_z_mu,
            prior_z_logvar,
            posterior_z_mu,
            posterior_z_logvar,
            prior_belief_conf,
            hidden_feature_gate,
            hidden_feature_used,
        )
