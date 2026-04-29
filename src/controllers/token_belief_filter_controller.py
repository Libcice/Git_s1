from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class TokenBeliefFilterMAC:
    """MAC for the token belief-filter path.

    Carries both Transformer memory and per-enemy online belief slots across
    timesteps. The controller stores only the latest tensors needed by the
    learner after each forward call.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.detach_recurrent_belief = getattr(args, "belief_detach_recurrent_state", True)

        self._setup_layout(scheme)
        self._build_agents(self.token_dim)

        self.hidden_states = None
        self.belief_mu = None
        self.belief_logvar = None
        self.belief_feat = None
        self.last_info = None

    def _setup_layout(self, scheme):
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
        self.tokens_per_step = 2 + self.args.enemy_num + (self.n_agents - 1)
        self.args.history_tokens_per_step = self.tokens_per_step

        self.enemy_state_feat_dim = 4 + self.unit_type_bits
        self.ally_state_feat_dim = 5 + self.unit_type_bits
        self.ally_state_dim = self.n_agents * self.ally_state_feat_dim
        self.enemy_state_dim = self.args.enemy_num * self.enemy_state_feat_dim
        self.args.enemy_obs_feat_dim = self.enemy_obs_feat_dim
        self.args.enemy_state_feat_dim = self.enemy_state_feat_dim
        self.args.enemy_state_dim = self.enemy_state_dim

    def _pad_tokens(self, tensor):
        pad = self.token_dim - tensor.size(-1)
        if pad <= 0:
            return tensor
        return th.nn.functional.pad(tensor, (0, pad))

    def _build_step_tokens(self, ep_batch, t):
        bs = ep_batch.batch_size
        raw_obs = ep_batch["obs"][:, t]
        cursor = 0

        move = raw_obs[:, :, cursor:cursor + self.move_dim]
        cursor += self.move_dim

        enemy_total = self.args.enemy_num * self.enemy_obs_feat_dim
        enemy = raw_obs[:, :, cursor:cursor + enemy_total].reshape(
            bs, self.n_agents, self.args.enemy_num, self.enemy_obs_feat_dim
        )
        cursor += enemy_total

        ally_total = (self.n_agents - 1) * self.ally_obs_feat_dim
        ally = raw_obs[:, :, cursor:cursor + ally_total].reshape(
            bs, self.n_agents, self.n_agents - 1, self.ally_obs_feat_dim
        )
        cursor += ally_total

        own = raw_obs[:, :, cursor:]
        extras = [own]
        if self.args.obs_last_action:
            if t == 0:
                prev_action = th.zeros_like(ep_batch["actions_onehot"][:, t])
            else:
                prev_action = ep_batch["actions_onehot"][:, t - 1]
            extras.append(prev_action)
        else:
            prev_action = th.zeros(bs, self.n_agents, self.args.n_actions, device=ep_batch.device)
        if self.args.obs_agent_id:
            extras.append(th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(bs, -1, -1))
        self_token = th.cat(extras, dim=-1)

        move_token = self._pad_tokens(move).unsqueeze(2)
        self_token_padded = self._pad_tokens(self_token).unsqueeze(2)
        enemy_tokens = self._pad_tokens(enemy)
        ally_tokens = self._pad_tokens(ally)
        step_tokens = th.cat([move_token, self_token_padded, enemy_tokens, ally_tokens], dim=2)

        current = {
            "move_token": self._pad_tokens(move).reshape(bs * self.n_agents, -1),
            "self_token": self._pad_tokens(self_token).reshape(bs * self.n_agents, -1),
            "enemy_tokens": self._pad_tokens(enemy).reshape(bs * self.n_agents, self.args.enemy_num, -1),
            "enemy_obs": enemy.reshape(bs * self.n_agents, self.args.enemy_num, -1),
            "enemy_visible": (enemy.abs().sum(dim=-1) > 0).reshape(bs * self.n_agents, self.args.enemy_num).float(),
            "ally_tokens": self._pad_tokens(ally).reshape(bs * self.n_agents, self.n_agents - 1, -1),
            "ally_visible": (ally.abs().sum(dim=-1) > 0).reshape(bs * self.n_agents, self.n_agents - 1).float(),
            "prev_action": prev_action.reshape(bs * self.n_agents, -1),
        }
        return step_tokens, current

    def _reshape_info(self, info, bs):
        reshaped = {}
        for key, value in info.items():
            if key in ("base_q", "belief_delta_q", "belief_gate"):
                reshaped[key] = value.view(bs, self.n_agents, -1)
            elif key in ("q_context", "visible_enemy_summary", "belief_confidence_summary"):
                reshaped[key] = value.view(bs, self.n_agents, -1)
            elif key == "belief_readout":
                reshaped[key] = value.view(bs, self.n_agents, self.args.n_actions, -1)
            elif key == "belief_attn":
                reshaped[key] = value.view(bs, self.n_agents, self.args.n_actions, self.args.enemy_num)
            elif key in (
                "prior_mu",
                "prior_logvar",
                "belief_mu",
                "belief_logvar",
                "future_mu",
                "future_logvar",
            ):
                reshaped[key] = value.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
            elif key in ("prior_feat", "belief_feat", "belief_confidence"):
                reshaped[key] = value.view(bs, self.n_agents, self.args.enemy_num, -1)
            elif key in ("enemy_visible", "correction_alpha"):
                reshaped[key] = value.view(bs, self.n_agents, self.args.enemy_num)
            else:
                reshaped[key] = value
        return reshaped

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def select_actions_vis(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, self.hidden_states, agent_outputs

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        step_tokens, current = self._build_step_tokens(ep_batch, t)
        step_tokens = step_tokens.reshape(bs * self.n_agents, -1, self.token_dim)
        hidden = self.hidden_states.reshape(bs * self.n_agents, -1)
        prev_mu = self.belief_mu.reshape(bs * self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        prev_logvar = self.belief_logvar.reshape(bs * self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        prev_feat = self.belief_feat.reshape(bs * self.n_agents, self.args.enemy_num, -1)
        if self.detach_recurrent_belief:
            prev_mu = prev_mu.detach()
            prev_logvar = prev_logvar.detach()
            prev_feat = prev_feat.detach()

        agent_outs, new_hidden, belief_mu, belief_logvar, belief_feat, info = self.agent(
            step_tokens,
            current,
            hidden,
            prev_mu,
            prev_logvar,
            prev_feat,
        )

        self.hidden_states = new_hidden.view(bs, self.n_agents, -1)
        self.last_info = self._reshape_info(info, bs)
        next_belief_mu = belief_mu.detach() if self.detach_recurrent_belief else belief_mu
        next_belief_logvar = belief_logvar.detach() if self.detach_recurrent_belief else belief_logvar
        next_belief_feat = belief_feat.detach() if self.detach_recurrent_belief else belief_feat
        self.belief_mu = next_belief_mu.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_logvar = next_belief_logvar.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_feat = next_belief_feat.view(bs, self.n_agents, self.args.enemy_num, -1)

        padded_belief = th.zeros(bs, self.n_agents, self.args.state_shape, device=ep_batch.device)
        flat_enemy_belief = self.belief_mu.reshape(bs, self.n_agents, -1)
        padded_belief[:, :, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim] = flat_enemy_belief
        if "belief" in ep_batch.data.transition_data:
            ep_batch["belief"][:, t] = padded_belief

        return agent_outs.view(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        mu, logvar, feat = self.agent.init_belief()
        self.belief_mu = mu.unsqueeze(0).expand(batch_size, self.n_agents, -1, -1).clone()
        self.belief_logvar = logvar.unsqueeze(0).expand(batch_size, self.n_agents, -1, -1).clone()
        self.belief_feat = feat.unsqueeze(0).expand(batch_size, self.n_agents, -1, -1).clone()
        self.last_info = None

    def get_belief(self):
        return self.belief_mu

    def get_belief_stats(self):
        if self.last_info is None:
            return self.belief_mu, self.belief_logvar, None
        return self.last_info["belief_mu"], self.last_info["belief_logvar"], None

    def get_belief_filter_info(self):
        return self.last_info

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
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
