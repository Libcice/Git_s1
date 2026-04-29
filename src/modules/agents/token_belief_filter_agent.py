import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBeliefFilterAgent(nn.Module):
    """Token Transformer + entity belief filter + action-conditioned delta-Q.

    The online path only consumes local observation tokens, previous action and
    the carried belief slots. Global state is intentionally not used here; it is
    only consumed by the learner for auxiliary targets.
    """

    def __init__(self, input_shape, args):
        super(TokenBeliefFilterAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.tokens_per_step = args.history_tokens_per_step
        self.n_enemies = args.enemy_num
        self.enemy_obs_feat_dim = args.enemy_obs_feat_dim
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.n_actions = args.n_actions
        self.use_belief_for_q = getattr(args, "use_belief_for_q", True)
        self.delta_gate_scale = getattr(args, "belief_delta_gate_scale", 0.5)
        self.logvar_min = getattr(args, "belief_logvar_min", -3.0)
        self.logvar_max = getattr(args, "belief_logvar_max", 1.0)

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
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

        self.prev_action_embed = nn.Linear(self.n_actions, self.hidden_dim)

        self.prev_belief_proj = nn.Sequential(
            nn.Linear(self.enemy_state_feat_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )

        prior_in_dim = self.hidden_dim * 4
        self.prior_feat_net = nn.Sequential(
            nn.Linear(prior_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prior_mu_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.prior_logvar_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)

        self.enemy_obs_proj = nn.Linear(self.enemy_obs_feat_dim, self.hidden_dim)
        self.enemy_obs_to_state = nn.Linear(self.enemy_obs_feat_dim, self.enemy_state_feat_dim)
        self.correction_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.correction_delta = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.enemy_state_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.visible_logvar = nn.Parameter(torch.full((self.enemy_state_feat_dim,), -2.5))

        future_in_dim = self.hidden_dim * 3
        self.future_feat_net = nn.Sequential(
            nn.Linear(future_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.future_mu_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.future_logvar_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)

        self.belief_state_proj = nn.Sequential(
            nn.Linear(self.enemy_state_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.q_context_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 5, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.base_q_head = nn.Linear(self.hidden_dim, self.n_actions)

        self.action_queries = nn.Parameter(torch.randn(self.n_actions, self.hidden_dim) * 0.02)
        self.action_context_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.belief_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.belief_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.delta_q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.delta_gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def init_belief(self):
        weight = self.token_embed.weight
        mu = weight.new_zeros(1, self.n_enemies, self.enemy_state_feat_dim)
        logvar = weight.new_zeros(1, self.n_enemies, self.enemy_state_feat_dim)
        feat = weight.new_zeros(1, self.n_enemies, self.hidden_dim)
        return mu, logvar, feat

    def _bounded_logvar(self, raw):
        return self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw)

    def _masked_mean(self, feats, mask, normalise_by_count=True):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        if normalise_by_count:
            denom = mask.sum(dim=1).clamp(min=1.0)
        else:
            denom = feats.new_tensor(float(max(1, feats.size(1))))
        return (feats * mask).sum(dim=1) / denom

    def _hidden_readout(self, q_context, belief_feat, hidden_mask, confidence=None):
        bs = q_context.size(0)
        query = self.action_queries.unsqueeze(0) + self.action_context_proj(q_context).unsqueeze(1)
        key = self.belief_key(belief_feat)
        value = self.belief_value(belief_feat)

        scores = torch.einsum("bah,bnh->ban", query, key) / math.sqrt(float(self.hidden_dim))
        mask = hidden_mask.float()
        scores = scores.masked_fill(mask.unsqueeze(1) <= 0.0, -1.0e9)
        weights = torch.softmax(scores, dim=-1) * mask.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        readout = torch.einsum("ban,bnh->bah", weights, value)

        hidden_frac = mask.mean(dim=1, keepdim=True)
        readout = readout * hidden_frac.unsqueeze(-1)
        hidden_present = mask.sum(dim=1, keepdim=True).gt(0).float()
        if confidence is None:
            confidence_summary = mask.sum(dim=1, keepdim=True).gt(0).float()
        else:
            conf_mask = mask
            confidence_summary = (confidence.squeeze(-1) * conf_mask).sum(dim=1, keepdim=True)
            confidence_summary = confidence_summary / conf_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        gate_input = torch.cat(
            [
                q_context.unsqueeze(1).expand(bs, self.n_actions, -1),
                readout,
                confidence_summary.unsqueeze(1).expand(bs, self.n_actions, -1),
            ],
            dim=-1,
        )
        gate = self.delta_gate_scale * torch.sigmoid(self.delta_gate_head(gate_input)).squeeze(-1)
        gate = gate * hidden_present
        delta_input = torch.cat(
            [q_context.unsqueeze(1).expand(bs, self.n_actions, -1), readout],
            dim=-1,
        )
        delta_q = self.delta_q_head(delta_input).squeeze(-1)
        return delta_q, gate, readout, weights, confidence_summary

    def compute_delta_q_from_slots(self, q_context, belief_feat, hidden_mask, confidence=None):
        delta_q, gate, readout, _, confidence_summary = self._hidden_readout(
            q_context, belief_feat, hidden_mask, confidence
        )
        return delta_q, gate, readout, confidence_summary

    def compute_oracle_delta_q(self, q_context, enemy_state, hidden_mask):
        oracle_feat = self.belief_state_proj(enemy_state)
        confidence = hidden_mask.unsqueeze(-1).float()
        return self.compute_delta_q_from_slots(q_context, oracle_feat, hidden_mask, confidence)

    def forward(self, step_tokens, current_step, hidden_state, prev_belief_mu, prev_belief_logvar, prev_belief_feat):
        step_emb = F.relu(self.token_embed(step_tokens))
        h_in = hidden_state.reshape(-1, self.hidden_dim).unsqueeze(1)

        seq = torch.cat([h_in, step_emb], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1)]
        encoded = self.encoder(seq)
        memory = encoded[:, 0, :]

        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))
        memory_slots = memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        action_slots = prev_action_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        prev_state_feat = self.prev_belief_proj(torch.cat([prev_belief_mu, prev_belief_logvar], dim=-1))
        prior_input = torch.cat([prev_belief_feat, prev_state_feat, memory_slots, action_slots], dim=-1)
        prior_feat = self.prior_feat_net(prior_input)
        prior_mu = self.prior_mu_head(prior_feat)
        prior_logvar = self._bounded_logvar(self.prior_logvar_head(prior_feat))

        enemy_obs_feat = F.relu(self.enemy_obs_proj(current_step["enemy_obs"]))
        obs_mu = self.enemy_obs_to_state(current_step["enemy_obs"])
        corr_gate_input = torch.cat([prior_feat, enemy_obs_feat], dim=-1)
        alpha = torch.sigmoid(self.correction_gate(corr_gate_input))
        alpha = alpha * current_step["enemy_visible"].unsqueeze(-1).float()

        visible_logvar = self.visible_logvar.clamp(self.logvar_min, self.logvar_max)
        visible_logvar = visible_logvar.view(1, 1, -1).expand_as(prior_logvar)
        belief_mu = prior_mu + alpha * (obs_mu - prior_mu)
        belief_logvar = prior_logvar + alpha * (visible_logvar - prior_logvar)
        corr_delta_input = torch.cat([prior_feat, enemy_obs_feat, obs_mu], dim=-1)
        belief_feat = prior_feat + alpha * self.correction_delta(corr_delta_input)

        future_input = torch.cat([belief_feat, memory_slots, action_slots], dim=-1)
        future_feat = self.future_feat_net(future_input)
        future_mu = self.future_mu_head(future_feat)
        future_logvar = self._bounded_logvar(self.future_logvar_head(future_feat))

        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        ally_feat = self._masked_mean(
            F.relu(self.token_embed(current_step["ally_tokens"])),
            current_step["ally_visible"],
            normalise_by_count=True,
        )
        visible_enemy_slots = F.relu(self.token_embed(current_step["enemy_tokens"]))
        visible_enemy_summary = self._masked_mean(
            visible_enemy_slots,
            current_step["enemy_visible"],
            normalise_by_count=False,
        )

        q_context = self.q_context_net(
            torch.cat([memory, move_feat, self_feat, ally_feat, visible_enemy_summary], dim=-1)
        )
        base_q = self.base_q_head(q_context)

        hidden_mask = 1.0 - current_step["enemy_visible"].float()
        confidence = torch.exp(-belief_logvar.mean(dim=-1, keepdim=True)).clamp(max=4.0) / 4.0
        belief_delta_q, belief_gate, belief_readout, attn_weights, confidence_summary = self._hidden_readout(
            q_context, belief_feat, hidden_mask, confidence
        )
        if self.use_belief_for_q:
            q = base_q + belief_gate * belief_delta_q
        else:
            q = base_q

        info = {
            "belief_mu": belief_mu,
            "belief_logvar": belief_logvar,
            "future_mu": future_mu,
            "future_logvar": future_logvar,
            # These tensors are only used for oracle targets/logging in the
            # learner, so avoid retaining their forward graph across the whole
            # episode. The actual Q graph is already retained through q/mac_out.
            "q_context": q_context.detach(),
            "base_q": base_q.detach(),
            "belief_delta_q": belief_delta_q,
            "belief_gate": belief_gate,
            "belief_confidence": confidence.detach(),
            "correction_alpha": alpha.squeeze(-1).detach(),
        }
        return q, memory, belief_mu, belief_logvar, belief_feat, info
