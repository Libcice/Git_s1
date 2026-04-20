import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .token_cvae_belief_agent import TokenCVAEBeliefAgent


class TokenValueCVAEBeliefAgent(TokenCVAEBeliefAgent):
    """Token-history CVAE agent with an auxiliary value-aware belief head."""

    def __init__(self, input_shape, args):
        super(TokenValueCVAEBeliefAgent, self).__init__(input_shape, args)
        self.belief_value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, args.n_actions),
        )

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
        real_enemy_feat = F.relu(self.real_enemy_proj(enemy_feat))
        q_context_raw = th.cat(
            [current_memory, move_feat, self_feat, ally_summary, prev_action_feat],
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
            prior_belief_conf = th.exp(
                -0.5
                * prior_belief_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max).mean(
                    dim=-1, keepdim=True
                )
            ).clamp(max=1.0)
        prior_belief_feat = F.relu(self.belief_enemy_proj(prior_belief_mu)) * prior_belief_conf

        visible_enemy_feat = current_step["enemy_visible"].unsqueeze(-1).float() * real_enemy_feat

        if self.use_belief_for_q:
            fused_enemy_feat = current_step["enemy_visible"].unsqueeze(-1).float() * real_enemy_feat + (
                1.0 - current_step["enemy_visible"].unsqueeze(-1).float()
            ) * prior_belief_feat
        else:
            fused_enemy_feat = visible_enemy_feat

        enemy_mask = th.ones_like(current_step["enemy_visible"])
        visible_enemy_summary = self._masked_mean(visible_enemy_feat, enemy_mask)
        enemy_summary = self._masked_mean(fused_enemy_feat, enemy_mask)
        enemy_attn = enemy_mask / enemy_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        q_visible = self.q_head(th.cat([q_context, visible_enemy_summary], dim=-1))
        q = self.q_head(th.cat([q_context, enemy_summary], dim=-1))

        q_context_expand = q_context.unsqueeze(1).expand(-1, self.n_enemies, -1)
        aux_belief_value = self.belief_value_head(th.cat([prior_belief_feat, q_context_expand], dim=-1))

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
            q_context,
            enemy_summary,
            fused_enemy_feat,
            enemy_attn,
            prior_belief_conf,
            real_enemy_feat,
            prior_belief_feat,
            aux_belief_value,
            q_visible,
        )
