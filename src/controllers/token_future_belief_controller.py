from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class TokenFutureBeliefMAC:
    """MAC for the token-history + future-belief path."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self._setup_layout(scheme)
        self._build_agents(self.token_dim)

        self.hidden_states = None
        self.current_prior_belief_mu = None
        self.current_prior_belief_logvar = None
        self.current_post_belief_mu = None
        self.current_post_belief_logvar = None
        self.current_prior_z_mu = None
        self.current_prior_z_logvar = None
        self.current_post_z_mu = None
        self.current_post_z_logvar = None
        self.base_q = None
        self.action_future_context = None
        self.future_action_repr = None
        self.selected_future_repr = None
        self.future_confidence = None
        self.future_gate = None
        self.future_delta_q = None
        self.future_weighted_delta_q = None
        self.current_prior_confidence = None
        self.current_base_gate = None

    def _setup_layout(self, scheme):
        del scheme
        env_args = getattr(self.args, "env_args", {})
        self.unit_type_bits = max(int(getattr(self.args, "unit_dim", 5)) - 5, 0)
        self.move_dim = 4
        if env_args.get("obs_pathing_grid", False):
            self.move_dim += 8
        if env_args.get("obs_terrain_height", False):
            self.move_dim += 9

        health_bits = 1 + 1 if env_args.get("obs_all_health", True) else 0
        self.enemy_obs_feat_dim = 4 + self.unit_type_bits + health_bits
        ally_health_bits = 1 + 1 if env_args.get("obs_all_health", True) else 0
        self.ally_obs_feat_dim = 4 + self.unit_type_bits + ally_health_bits
        if env_args.get("obs_last_action", False):
            self.ally_obs_feat_dim += self.args.n_actions

        self.raw_own_dim = (
            self.args.obs_shape
            - self.move_dim
            - self.args.enemy_num * self.enemy_obs_feat_dim
            - (self.n_agents - 1) * self.ally_obs_feat_dim
        )
        self.self_token_raw_dim = self.raw_own_dim
        if self.args.obs_last_action:
            self.self_token_raw_dim += self.args.n_actions
        if self.args.obs_agent_id:
            self.self_token_raw_dim += self.n_agents

        self.token_dim = max(
            self.move_dim,
            self.enemy_obs_feat_dim,
            self.ally_obs_feat_dim,
            self.self_token_raw_dim,
        )

        self.enemy_state_feat_dim = 4 + self.unit_type_bits
        self.ally_state_feat_dim = 5 + self.unit_type_bits
        self.ally_state_dim = self.n_agents * self.ally_state_feat_dim
        self.enemy_state_dim = self.args.enemy_num * self.enemy_state_feat_dim
        self.args.enemy_state_feat_dim = self.enemy_state_feat_dim
        self.args.enemy_state_dim = self.enemy_state_dim

    def _pad_tokens(self, tensor):
        pad = self.token_dim - tensor.size(-1)
        if pad <= 0:
            return tensor
        return th.nn.functional.pad(tensor, (0, pad))

    def _build_step_inputs(self, ep_batch, t):
        bs = ep_batch.batch_size
        raw_obs = ep_batch["obs"][:, t]
        cursor = 0

        move = raw_obs[:, :, cursor:cursor + self.move_dim]
        cursor += self.move_dim

        enemy_total = self.args.enemy_num * self.enemy_obs_feat_dim
        enemy = raw_obs[:, :, cursor:cursor + enemy_total].reshape(
            bs,
            self.n_agents,
            self.args.enemy_num,
            self.enemy_obs_feat_dim,
        )
        cursor += enemy_total

        ally_total = (self.n_agents - 1) * self.ally_obs_feat_dim
        ally = raw_obs[:, :, cursor:cursor + ally_total].reshape(
            bs,
            self.n_agents,
            self.n_agents - 1,
            self.ally_obs_feat_dim,
        )
        cursor += ally_total

        own = raw_obs[:, :, cursor:]
        extras = [own]
        if self.args.obs_last_action:
            if t == 0:
                extras.append(th.zeros_like(ep_batch["actions_onehot"][:, t]))
            else:
                extras.append(ep_batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            agent_ids = th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(bs, -1, -1)
            extras.append(agent_ids)
        self_token = th.cat(extras, dim=-1)

        if t == 0:
            prev_action = th.zeros_like(ep_batch["actions_onehot"][:, t])
        else:
            prev_action = ep_batch["actions_onehot"][:, t - 1]

        current = {
            "move_token": self._pad_tokens(move).reshape(bs * self.n_agents, -1),
            "self_token": self._pad_tokens(self_token).reshape(bs * self.n_agents, -1),
            "enemy_tokens": self._pad_tokens(enemy).reshape(bs * self.n_agents, self.args.enemy_num, -1),
            "enemy_visible": (enemy.abs().sum(dim=-1) > 0).reshape(bs * self.n_agents, self.args.enemy_num).float(),
            "ally_tokens": self._pad_tokens(ally).reshape(bs * self.n_agents, self.n_agents - 1, -1),
            "ally_visible": (ally.abs().sum(dim=-1) > 0).reshape(bs * self.n_agents, self.n_agents - 1).float(),
            "prev_action": prev_action.reshape(bs * self.n_agents, -1),
        }
        return current

    def _view_aux(self, aux, key, shape):
        value = aux.get(key)
        if value is None:
            return None
        return value.view(*shape)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def select_actions_vis(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, self.hidden_states, agent_outputs

    def forward(
        self,
        ep_batch,
        t,
        hidden_enemy_state=None,
        next_hidden_enemy_state=None,
        selected_actions=None,
        test_mode=False,
    ):
        del test_mode
        bs = ep_batch.batch_size
        current = self._build_step_inputs(ep_batch, t)
        hidden = self.hidden_states.reshape(bs * self.n_agents, -1)
        flat_selected_actions = None
        if selected_actions is not None:
            flat_selected_actions = selected_actions.reshape(bs * self.n_agents, -1)
        agent_outs, new_hidden, aux = self.agent(
            current,
            hidden,
            hidden_enemy_state=hidden_enemy_state,
            next_hidden_enemy_state=next_hidden_enemy_state,
            selected_actions=flat_selected_actions,
        )

        self.hidden_states = new_hidden.view(bs, self.n_agents, -1)
        belief_shape = (bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        latent_shape = (bs, self.n_agents, -1)
        gate_shape = (bs, self.n_agents, self.args.enemy_num, 1)
        selected_repr_shape = (bs, self.n_agents, self.args.enemy_num, self.agent.future_dim)
        action_q_shape = (bs, self.n_agents, self.args.n_actions)

        self.current_prior_belief_mu = self._view_aux(aux, "current_prior_mu", belief_shape)
        self.current_prior_belief_logvar = self._view_aux(aux, "current_prior_logvar", belief_shape)
        self.current_post_belief_mu = self._view_aux(aux, "current_post_mu", belief_shape)
        self.current_post_belief_logvar = self._view_aux(aux, "current_post_logvar", belief_shape)
        self.current_prior_z_mu = self._view_aux(aux, "current_prior_z_mu", latent_shape)
        self.current_prior_z_logvar = self._view_aux(aux, "current_prior_z_logvar", latent_shape)
        self.current_post_z_mu = self._view_aux(aux, "current_post_z_mu", latent_shape)
        self.current_post_z_logvar = self._view_aux(aux, "current_post_z_logvar", latent_shape)

        self.current_prior_confidence = self._view_aux(aux, "current_prior_conf", gate_shape)
        self.current_base_gate = self._view_aux(aux, "current_base_gate", gate_shape)
        self.base_q = self._view_aux(aux, "base_q", action_q_shape)
        self.action_future_context = None
        self.future_action_repr = None
        self.selected_future_repr = self._view_aux(aux, "selected_future_repr", selected_repr_shape)
        self.future_confidence = None
        self.future_gate = self._view_aux(aux, "future_gate", action_q_shape)
        self.future_delta_q = self._view_aux(aux, "future_delta_q", action_q_shape)
        self.future_weighted_delta_q = self._view_aux(aux, "future_weighted_delta_q", action_q_shape)

        padded_belief = th.zeros(bs, self.n_agents, self.args.state_shape, device=ep_batch.device)
        flat_enemy_belief = self.current_prior_belief_mu.reshape(bs, self.n_agents, -1)
        padded_belief[:, :, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim] = flat_enemy_belief
        if "belief" in ep_batch.data.transition_data:
            ep_batch["belief"][:, t] = padded_belief

        return agent_outs.view(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        device = self.agent.init_hidden().device
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).clone()
        self.current_prior_belief_mu = None
        self.current_prior_belief_logvar = None
        self.current_post_belief_mu = None
        self.current_post_belief_logvar = None
        self.current_prior_z_mu = None
        self.current_prior_z_logvar = None
        self.current_post_z_mu = None
        self.current_post_z_logvar = None
        self.base_q = None
        self.action_future_context = None
        self.future_action_repr = None
        self.selected_future_repr = None
        self.future_confidence = None
        self.future_gate = None
        self.future_delta_q = None
        self.future_weighted_delta_q = None
        self.current_prior_confidence = None
        self.current_base_gate = None
        del device

    def set_future_q_alpha(self, alpha):
        if hasattr(self.agent, "set_future_q_alpha"):
            self.agent.set_future_q_alpha(alpha)

    def set_belief_q_alpha(self, alpha):
        self.set_future_q_alpha(alpha)

    def get_belief(self):
        return self.current_prior_belief_mu

    def get_current_prior_belief_stats(self):
        return self.current_prior_belief_mu, self.current_prior_belief_logvar

    def get_current_posterior_belief_stats(self):
        return self.current_post_belief_mu, self.current_post_belief_logvar

    def get_current_prior_latent_stats(self):
        return self.current_prior_z_mu, self.current_prior_z_logvar

    def get_current_posterior_latent_stats(self):
        return self.current_post_z_mu, self.current_post_z_logvar

    def get_next_prior_belief_stats(self):
        return None, None

    def get_next_posterior_belief_stats(self):
        return None, None

    def get_next_prior_latent_stats(self):
        return None, None

    def get_next_posterior_latent_stats(self):
        return None, None

    def get_behavior_residual_feat(self):
        return None

    def get_current_prior_confidence(self):
        return self.current_prior_confidence

    def get_current_base_gate(self):
        return self.current_base_gate

    def get_next_prior_confidence(self):
        return None

    def get_future_res_gate(self):
        return self.future_gate

    def get_future_res_feat(self):
        return self.selected_future_repr

    def get_future_action_res_feat(self):
        return self.selected_future_repr

    def get_base_q(self):
        return self.base_q

    def get_action_future_context(self):
        return self.action_future_context

    def get_future_action_repr(self):
        return None

    def get_selected_future_repr(self):
        return self.selected_future_repr

    def get_future_confidence(self):
        return self.future_confidence

    def get_future_gate(self):
        return self.future_gate

    def get_future_delta_q(self):
        return self.future_delta_q

    def get_future_weighted_delta_q(self):
        return self.future_weighted_delta_q

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        if self.args.agent in agent_REGISTRY:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
            return

        if self.args.agent != "token_future_belief":
            raise KeyError("Unknown future belief agent '{}'".format(self.args.agent))

        from modules.agents.token_future_belief_agent import TokenFutureBeliefAgent

        self.agent = TokenFutureBeliefAgent(input_shape, self.args)
