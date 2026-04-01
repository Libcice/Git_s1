from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class BeliefSlotTransformerMAC:
    """MAC for explicit belief-slot transformer experiments.

    This path keeps a recurrent memory state and an explicit belief slot for
    each enemy. The agent receives only current observation tokens plus the
    previous latent state, rather than a long explicit history window.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self._setup_layout(scheme)
        self._build_agents(self.token_dim)

        self.hidden_states = None
        self.belief_slots = None
        self.belief_mu = None
        self.belief_logvar = None
        self.belief_u = None
        self.unseen_steps = None

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

        self.raw_own_dim = self.args.obs_shape - self.move_dim - self.args.enemy_num * self.enemy_obs_feat_dim - (self.n_agents - 1) * self.ally_obs_feat_dim
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
        self.ally_state_feat_dim = self.n_agents * (5 + self.unit_type_bits)
        self.enemy_state_dim = self.args.enemy_num * self.enemy_state_feat_dim
        self.args.enemy_state_feat_dim = self.enemy_state_feat_dim
        self.args.enemy_state_dim = self.enemy_state_dim

    def _pad_tokens(self, tensor):
        pad = self.token_dim - tensor.size(-1)
        if pad <= 0:
            return tensor
        return th.nn.functional.pad(tensor, (0, pad))

    def _build_current_inputs(self, ep_batch, t):
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
        if t == 0:
            prev_action = th.zeros_like(ep_batch["actions_onehot"][:, t])
        else:
            prev_action = ep_batch["actions_onehot"][:, t - 1]

        extras = [own]
        if self.args.obs_last_action:
            extras.append(prev_action)
        if self.args.obs_agent_id:
            extras.append(th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(bs, -1, -1))
        self_token = th.cat(extras, dim=-1)

        enemy_visible = (enemy.abs().sum(dim=-1) > 0).float()
        ally_visible = (ally.abs().sum(dim=-1) > 0).float()

        if self.unseen_steps is None:
            prev_unseen = enemy_visible.new_zeros(bs, self.n_agents, self.args.enemy_num)
        else:
            prev_unseen = self.unseen_steps
        current_unseen = (1.0 - enemy_visible) * (prev_unseen + 1.0)

        current = {
            "move_token": self._pad_tokens(move).reshape(bs * self.n_agents, -1),
            "self_token": self._pad_tokens(self_token).reshape(bs * self.n_agents, -1),
            "enemy_tokens": self._pad_tokens(enemy).reshape(bs * self.n_agents, self.args.enemy_num, -1),
            "enemy_visible": enemy_visible.reshape(bs * self.n_agents, self.args.enemy_num),
            "unseen_steps": current_unseen.reshape(bs * self.n_agents, self.args.enemy_num),
            "ally_tokens": self._pad_tokens(ally).reshape(bs * self.n_agents, self.n_agents - 1, -1),
            "ally_visible": ally_visible.reshape(bs * self.n_agents, self.n_agents - 1),
            "prev_action": prev_action.reshape(bs * self.n_agents, -1),
        }
        return current, current_unseen

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
        current, current_unseen = self._build_current_inputs(ep_batch, t)
        prev_memory = self.hidden_states.reshape(bs * self.n_agents, -1)
        prev_belief_slots = self.belief_slots.reshape(bs * self.n_agents, self.args.enemy_num, -1)

        agent_outs, new_memory, new_slots, belief_mu, belief_logvar, belief_u = self.agent(
            current,
            prev_belief_slots,
            prev_memory,
        )

        self.hidden_states = new_memory.view(bs, self.n_agents, -1)
        self.belief_slots = new_slots.view(bs, self.n_agents, self.args.enemy_num, -1)
        self.belief_mu = belief_mu.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_logvar = belief_logvar.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_u = belief_u.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim, -1)
        # Track how long each enemy has remained hidden for each local agent.
        self.unseen_steps = current_unseen.clone()

        padded_belief = th.zeros(bs, self.n_agents, self.args.state_shape, device=ep_batch.device)
        flat_enemy_belief = self.belief_mu.reshape(bs, self.n_agents, -1)
        padded_belief[:, :, self.ally_state_feat_dim:self.ally_state_feat_dim + self.enemy_state_dim] = flat_enemy_belief
        ep_batch["belief"][:, t] = padded_belief

        return agent_outs.view(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        init_slots = self.agent.init_belief().unsqueeze(0).expand(batch_size, self.n_agents, -1, -1)
        self.belief_slots = init_slots.clone()
        self.belief_mu = None
        self.belief_logvar = None
        self.belief_u = None
        self.unseen_steps = self.hidden_states.new_zeros(batch_size, self.n_agents, self.args.enemy_num)

    def get_belief(self):
        return self.belief_mu

    def get_belief_stats(self):
        return self.belief_mu, self.belief_logvar, self.belief_u

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
