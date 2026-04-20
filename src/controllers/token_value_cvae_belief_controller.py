from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .token_cvae_belief_controller import TokenCVAEBeliefMAC


class TokenValueCVAEBeliefMAC(TokenCVAEBeliefMAC):
    """MAC for token-history CVAE belief with auxiliary value distillation."""

    def __init__(self, scheme, groups, args):
        super(TokenValueCVAEBeliefMAC, self).__init__(scheme, groups, args)
        self.q = None
        self.current_memory = None
        self.real_enemy_feat = None
        self.prior_belief_feat = None
        self.aux_belief_values = None
        self.q_visible = None

    def forward(self, ep_batch, t, hidden_enemy_state=None, test_mode=False):
        bs = ep_batch.batch_size
        current = self._build_step_inputs(ep_batch, t)
        hidden = self.hidden_states.reshape(bs * self.n_agents, -1)
        (
            agent_outs,
            new_hidden,
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
            fused_enemy_feats,
            enemy_attn,
            prior_belief_confidence,
            real_enemy_feat,
            prior_belief_feat,
            aux_belief_values,
            q_visible,
        ) = self.agent(current, hidden, hidden_enemy_state=hidden_enemy_state)

        self.hidden_states = new_hidden.view(bs, self.n_agents, -1)
        self.current_memory = self.hidden_states
        self.q = agent_outs.view(bs, self.n_agents, -1)
        self.prior_belief_mu = prior_belief_mu.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.prior_belief_logvar = prior_belief_logvar.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.prior_z_mu = prior_z_mu.view(bs, self.n_agents, -1)
        self.prior_z_logvar = prior_z_logvar.view(bs, self.n_agents, -1)
        self.q_context = q_context.view(bs, self.n_agents, -1)
        self.enemy_summary = enemy_summary.view(bs, self.n_agents, -1)
        self.fused_enemy_feats = fused_enemy_feats.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.enemy_attn = enemy_attn.view(bs, self.n_agents, self.args.enemy_num)
        self.prior_belief_confidence = prior_belief_confidence.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.real_enemy_feat = real_enemy_feat.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.prior_belief_feat = prior_belief_feat.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.aux_belief_values = aux_belief_values.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.q_visible = q_visible.view(bs, self.n_agents, -1)

        if posterior_belief_mu is not None:
            self.posterior_belief_mu = posterior_belief_mu.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
            self.posterior_belief_logvar = posterior_belief_logvar.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
            self.posterior_z_mu = posterior_z_mu.view(bs, self.n_agents, -1)
            self.posterior_z_logvar = posterior_z_logvar.view(bs, self.n_agents, -1)
        else:
            self.posterior_belief_mu = None
            self.posterior_belief_logvar = None
            self.posterior_z_mu = None
            self.posterior_z_logvar = None

        padded_belief = th.zeros(bs, self.n_agents, self.args.state_shape, device=ep_batch.device)
        flat_enemy_belief = self.prior_belief_mu.reshape(bs, self.n_agents, -1)
        padded_belief[:, :, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim] = flat_enemy_belief
        ep_batch["belief"][:, t] = padded_belief

        return self.q

    def init_hidden(self, batch_size):
        device = self.agent.init_hidden().device
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).clone()
        self.current_memory = self.hidden_states
        self.q = None
        self.prior_belief_mu = None
        self.prior_belief_logvar = None
        self.posterior_belief_mu = None
        self.posterior_belief_logvar = None
        self.prior_z_mu = None
        self.prior_z_logvar = None
        self.posterior_z_mu = None
        self.posterior_z_logvar = None
        self.q_context = None
        self.enemy_summary = None
        self.fused_enemy_feats = None
        self.enemy_attn = None
        self.prior_belief_confidence = None
        self.real_enemy_feat = None
        self.prior_belief_feat = None
        self.aux_belief_values = None
        self.q_visible = None

    def get_q(self):
        return self.q

    def get_current_memory(self):
        return self.current_memory

    def get_real_enemy_feat(self):
        return self.real_enemy_feat

    def get_prior_belief_feat(self):
        return self.prior_belief_feat

    def get_aux_belief_values(self):
        return self.aux_belief_values

    def get_q_visible(self):
        return self.q_visible
