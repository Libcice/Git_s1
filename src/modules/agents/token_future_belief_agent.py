import math

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


def _init_last_bias(module, value):
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, value)
            return


class TokenFutureBeliefAgent(nn.Module):
    """Token-history agent with current CVAE belief and action-conditioned future residuals."""

    def __init__(self, input_shape, args):
        super(TokenFutureBeliefAgent, self).__init__()
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
        self.future_dim = int(getattr(args, "future_repr_dim", max(16, self.hidden_dim // 3)))
        self.disable_belief = getattr(args, "disable_belief", False)
        self.use_belief_for_q = getattr(args, "use_belief_for_q", True) and not self.disable_belief
        self.belief_prior_type = getattr(args, "belief_prior_type", "conditional")
        self.belief_logvar_min = getattr(args, "belief_logvar_min", -3.0)
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
        self.current_base_gate_floor = float(getattr(args, "current_base_gate_floor", 0.1))
        self.belief_alive_gate_floor = float(getattr(args, "belief_alive_gate_floor", 0.05))
        self.future_gate_floor = float(getattr(args, "future_gate_floor", 0.0))
        self.future_q_alpha = 0.0

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

        self.real_enemy_proj = _build_mlp(self.hidden_dim * 2, self.hidden_dim * 2, self.hidden_dim)
        self.current_belief_feat_proj = _build_mlp(self.enemy_state_feat_dim, self.hidden_dim, self.hidden_dim)

        self.visible_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.visible_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.visible_value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.belief_query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.belief_key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.belief_value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.q_base_proj = _build_mlp(self.hidden_dim * 3, self.hidden_dim * 2, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim, args.n_actions)

        self.future_context_proj = _build_mlp(self.hidden_dim * 2, self.hidden_dim, self.future_dim)
        self.future_action_embed = nn.Linear(args.n_actions, self.future_dim)
        self.future_baseq_proj = nn.Linear(1, self.future_dim)
        self.future_repr_head = _build_mlp(self.hidden_dim * 3, self.hidden_dim * 2, self.future_dim)
        self.selected_future_repr_head = _build_mlp(self.future_dim * 2, self.hidden_dim, self.future_dim)
        self.future_target_proj = nn.Linear(self.hidden_dim, self.future_dim)
        self.future_conf_head = _build_mlp(self.future_dim * 2, self.hidden_dim, 1)
        self.future_query_proj = nn.Linear(self.future_dim, self.future_dim)
        self.future_key_proj = nn.Linear(self.future_dim, self.future_dim)
        self.future_value_proj = nn.Linear(self.future_dim, self.future_dim)
        self.future_gate_proj = _build_mlp(self.future_dim + 4, self.hidden_dim, 1)
        self.future_delta_head = _build_mlp(self.future_dim * 2 + 1, self.hidden_dim, 1)
        _init_last_bias(self.future_gate_proj, float(getattr(args, "future_gate_init_bias", -4.0)))
        _init_last_bias(self.future_delta_head, 0.0)

        if self.belief_prior_type == "conditional":
            self.current_prior_encoder = _build_mlp(self.hidden_dim, self.hidden_dim * 2, self.latent_dim * 2)
        else:
            self.current_prior_encoder = None

        self.current_posterior_encoder = _build_mlp(
            self.hidden_dim + self.n_enemies * self.enemy_state_feat_dim,
            self.hidden_dim * 2,
            self.latent_dim * 2,
        )
        self.current_decoder_trunk = _build_mlp(
            self.hidden_dim + self.latent_dim,
            self.hidden_dim * 2,
            self.hidden_dim * 2,
        )
        self.current_decoder_mu = nn.Linear(self.hidden_dim * 2, self.n_enemies * self.enemy_state_feat_dim)
        self.current_decoder_logvar = nn.Linear(self.hidden_dim * 2, self.n_enemies * self.enemy_state_feat_dim)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def set_future_q_alpha(self, alpha):
        self.future_q_alpha = float(alpha)

    def set_belief_q_alpha(self, alpha):
        self.set_future_q_alpha(alpha)

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _reparameterize(self, mu, logvar):
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

    def _belief_confidence(self, belief_logvar):
        avg_logvar = belief_logvar.clamp(
            min=self.belief_logvar_min,
            max=self.belief_logvar_max,
        ).mean(dim=-1, keepdim=True)
        return th.sigmoid((self.belief_confidence_center - avg_logvar) / self.belief_confidence_temp)

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

    def _prior_latent(self, context):
        if self.belief_prior_type == "standard_normal":
            prior_mu = context.new_zeros(context.size(0), self.latent_dim)
            prior_logvar = context.new_zeros(context.size(0), self.latent_dim)
        else:
            prior_stats = self.current_prior_encoder(context)
            prior_mu, prior_logvar = self._split_bounded_stats(prior_stats)
        return prior_mu, prior_logvar

    def _posterior_latent(self, context, hidden_enemy_state):
        hidden_flat = hidden_enemy_state.reshape(hidden_enemy_state.size(0), -1)
        posterior_stats = self.current_posterior_encoder(th.cat([context, hidden_flat], dim=-1))
        posterior_mu, posterior_logvar = self._split_bounded_stats(posterior_stats)
        return posterior_mu, posterior_logvar

    def _decode_current_belief(self, context, latent_z):
        decoder_hidden = self.current_decoder_trunk(th.cat([context, latent_z], dim=-1))
        belief_mu = self.current_decoder_mu(decoder_hidden).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        belief_logvar = self._bound_logvar(
            self.current_decoder_logvar(decoder_hidden).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        )
        return belief_mu, belief_logvar

    def _cross_attention_readout(self, query, slots, slot_mask, key_proj, value_proj):
        if slots.size(1) == 0:
            return query.new_zeros(query.size(0), self.hidden_dim)

        keys = key_proj(slots)
        values = value_proj(slots)
        scores = th.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / math.sqrt(keys.size(-1))
        mask_bool = slot_mask > 0
        scores = scores.masked_fill(~mask_bool, -1e9)
        attn = th.softmax(scores, dim=-1)
        attn = attn * mask_bool.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return th.bmm(attn.unsqueeze(1), values).squeeze(1)

    def _action_cross_attention_readout(self, query, slots, slot_mask):
        if slots.size(1) == 0:
            return query.new_zeros(query.size(0), query.size(1), self.future_dim)

        keys = self.future_key_proj(slots)
        values = self.future_value_proj(slots)
        q = self.future_query_proj(query)
        scores = th.matmul(q, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))
        mask_bool = (slot_mask > 0).unsqueeze(1)
        scores = scores.masked_fill(~mask_bool, -1e9)
        attn = th.softmax(scores, dim=-1)
        attn = attn * mask_bool.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return th.bmm(attn, values)

    def forward(
        self,
        current_step,
        prev_memory,
        hidden_enemy_state=None,
        next_hidden_enemy_state=None,
        selected_actions=None,
    ):
        del next_hidden_enemy_state
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

        del ally_ctx

        raw_ally_summary = self._masked_mean(ally_feat, current_step["ally_visible"])
        visible_mask = current_step["enemy_visible"].float().unsqueeze(-1)
        hidden_mask = 1.0 - visible_mask
        hidden_mask_flat = hidden_mask.squeeze(-1)

        real_enemy_feat = self.real_enemy_proj(th.cat([enemy_feat, enemy_ctx], dim=-1))
        visible_enemy_slots = visible_mask * real_enemy_feat

        q_context = self.q_context(
            th.cat([current_memory, move_feat, self_feat, raw_ally_summary, prev_action_feat], dim=-1)
        )
        history_context = self.history_context(th.cat([current_memory, prev_action_feat], dim=-1))

        current_prior_z_mu, current_prior_z_logvar = self._prior_latent(history_context)
        current_prior_mu, current_prior_logvar = self._decode_current_belief(history_context, current_prior_z_mu)
        if self.belief_prior_type == "standard_normal":
            current_prior_conf = current_prior_mu.new_ones(current_prior_mu.size(0), self.n_enemies, 1)
        else:
            current_prior_conf = self._belief_confidence(current_prior_logvar)
        current_belief_feat = F.relu(self.current_belief_feat_proj(current_prior_mu))

        # The online policy cannot observe the state alive mask. Use the predicted
        # first enemy-state feature as a soft validity gate so dead invisible slots
        # do not keep acting like ordinary hidden enemies.
        pred_alive_gate = current_prior_mu[..., :1].clamp(min=0.0, max=1.0)
        pred_hidden_alive = hidden_mask * pred_alive_gate
        hidden_entity_gate = hidden_mask * (
            self.belief_alive_gate_floor + (1.0 - self.belief_alive_gate_floor) * pred_alive_gate
        )
        hidden_entity_mask_flat = hidden_entity_gate.squeeze(-1)

        current_base_gate = hidden_entity_gate * (
            self.current_base_gate_floor + (1.0 - self.current_base_gate_floor) * current_prior_conf
        )

        if self.use_belief_for_q:
            hidden_enemy_slots = current_base_gate * current_belief_feat
        else:
            hidden_enemy_slots = current_belief_feat.new_zeros(current_belief_feat.shape)

        visible_readout = self._cross_attention_readout(
            self.visible_query_proj(q_context),
            visible_enemy_slots,
            visible_mask.squeeze(-1),
            self.visible_key_proj,
            self.visible_value_proj,
        )
        belief_readout = self._cross_attention_readout(
            self.belief_query_proj(q_context),
            hidden_enemy_slots,
            hidden_entity_mask_flat,
            self.belief_key_proj,
            self.belief_value_proj,
        )
        belief_active = (pred_hidden_alive.sum(dim=1) > 1e-6).float()
        belief_readout = belief_active * belief_readout
        if not self.use_belief_for_q:
            belief_readout = q_context.new_zeros(q_context.size(0), self.hidden_dim)

        q_base_feat = self.q_base_proj(th.cat([q_context, visible_readout, belief_readout], dim=-1))
        base_q = self.q_head(q_base_feat)

        action_eye = th.eye(self.args.n_actions, device=q_context.device)
        action_feat = F.relu(self.future_action_embed(action_eye)).unsqueeze(0).expand(q_context.size(0), -1, -1)
        future_context_base = self.future_context_proj(th.cat([q_context, q_base_feat], dim=-1)).unsqueeze(1)
        action_future_context = F.relu(
            future_context_base + action_feat + self.future_baseq_proj(base_q.unsqueeze(-1))
        )

        future_slot_context = q_context.unsqueeze(1).expand(-1, self.n_enemies, -1)
        future_entity_slots = self.future_repr_head(
            th.cat([future_slot_context, current_belief_feat, real_enemy_feat], dim=-1)
        )
        future_readout = self._action_cross_attention_readout(
            action_future_context,
            future_entity_slots,
            hidden_entity_mask_flat,
        )
        future_conf_summary = th.sigmoid(self.future_conf_head(th.cat([future_readout, action_future_context], dim=-1)))

        hidden_frac = pred_hidden_alive.squeeze(-1).mean(dim=-1, keepdim=True)
        current_conf_summary = self._masked_mean(current_prior_conf, hidden_entity_mask_flat)
        gate_input = th.cat(
            [
                action_future_context,
                base_q.unsqueeze(-1),
                current_conf_summary.unsqueeze(1).expand(-1, self.args.n_actions, -1),
                future_conf_summary,
                hidden_frac.unsqueeze(1).expand(-1, self.args.n_actions, -1),
            ],
            dim=-1,
        )
        future_gate = self.future_gate_floor + (1.0 - self.future_gate_floor) * th.sigmoid(
            self.future_gate_proj(gate_input).squeeze(-1)
        )
        future_gate = future_gate * belief_active.expand_as(future_gate)

        future_delta_q = self.future_delta_head(
            th.cat([action_future_context, future_readout, base_q.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)

        selected_future_repr = None
        if selected_actions is not None:
            selected_index = selected_actions.long().reshape(-1, 1, 1).expand(-1, 1, self.future_dim)
            selected_future_context = th.gather(action_future_context, dim=1, index=selected_index).squeeze(1)
            selected_future_context = selected_future_context.unsqueeze(1).expand(-1, self.n_enemies, -1)
            selected_future_repr = self.selected_future_repr_head(
                th.cat([selected_future_context, future_entity_slots], dim=-1)
            )

        if not self.use_belief_for_q:
            future_gate = future_gate.new_zeros(future_gate.shape)
            future_delta_q = future_delta_q.new_zeros(future_delta_q.shape)
            future_entity_slots = future_entity_slots.new_zeros(future_entity_slots.shape)
            if selected_future_repr is not None:
                selected_future_repr = selected_future_repr.new_zeros(selected_future_repr.shape)

        q = base_q + self.future_q_alpha * future_gate * future_delta_q

        current_post_mu = None
        current_post_logvar = None
        current_post_z_mu = None
        current_post_z_logvar = None
        if hidden_enemy_state is not None:
            current_post_z_mu, current_post_z_logvar = self._posterior_latent(
                history_context,
                hidden_enemy_state,
            )
            current_post_z = self._reparameterize(current_post_z_mu, current_post_z_logvar)
            current_post_mu, current_post_logvar = self._decode_current_belief(history_context, current_post_z)

        aux = {
            "current_prior_mu": current_prior_mu,
            "current_prior_logvar": current_prior_logvar,
            "current_post_mu": current_post_mu,
            "current_post_logvar": current_post_logvar,
            "current_prior_z_mu": current_prior_z_mu,
            "current_prior_z_logvar": current_prior_z_logvar,
            "current_post_z_mu": current_post_z_mu,
            "current_post_z_logvar": current_post_z_logvar,
            "current_prior_conf": current_prior_conf,
            "current_base_gate": current_base_gate,
            "base_q": base_q,
            "future_entity_slots": future_entity_slots,
            "selected_future_repr": selected_future_repr,
            "future_gate": future_gate,
            "future_delta_q": future_delta_q,
            "future_weighted_delta_q": self.future_q_alpha * future_gate * future_delta_q,
        }
        return q, current_memory, aux
