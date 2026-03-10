from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class HistoryTokenRNNBeliefMAC:
    """MAC for history-token RNN belief experiments.

    This path is isolated from the original BasicMAC so the default
    QMIX+RNN implementation remains unchanged.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self._setup_layout(scheme)
        self._build_agents(self.token_dim)

        self.hidden_states = None
        self.history_tokens = None
        self.belief_mu = None
        self.belief_logvar = None
        self.belief_u = None

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
        self.tokens_per_step = 2 + self.args.enemy_num + (self.n_agents - 1)
        self.args.history_tokens_per_step = self.tokens_per_step

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
                extras.append(th.zeros_like(ep_batch["actions_onehot"][:, t]))
            else:
                extras.append(ep_batch["actions_onehot"][:, t - 1])
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
            # Enemy rows are zero when the unit is outside sight or dead.
            "enemy_visible": (enemy.abs().sum(dim=-1) > 0).reshape(bs * self.n_agents, self.args.enemy_num).float(),
        }
        return step_tokens, current

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
        step_tokens = step_tokens.unsqueeze(2)

        # [PATCH] Use only past steps as RNN history.
        # [PATCH] Current-step features are provided through `current`.
        if self.history_tokens is None:
            history_tokens = step_tokens[:, :, :0]
        else:
            history_tokens = self.history_tokens

        history = history_tokens.reshape(bs * self.n_agents, -1, self.token_dim)
        hidden = self.hidden_states.reshape(bs * self.n_agents, -1)
        agent_outs, new_hidden, belief_mu, belief_logvar, belief_u = self.agent(history, current, hidden)

        # [PATCH] Push current step after forward, so no same-step duplication.
        if self.history_tokens is None:
            self.history_tokens = step_tokens
        else:
            self.history_tokens = th.cat([self.history_tokens, step_tokens], dim=2)

        # [PATCH] Support history_steps=0: keep an empty history buffer.
        history_keep = int(getattr(self.args, "history_steps", 4))
        if history_keep <= 0:
            self.history_tokens = self.history_tokens[:, :, :0]
        else:
            self.history_tokens = self.history_tokens[:, :, -history_keep:]

        self.hidden_states = new_hidden.view(bs, self.n_agents, -1)
        self.belief_mu = belief_mu.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_logvar = belief_logvar.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim)
        self.belief_u = belief_u.view(bs, self.n_agents, self.args.enemy_num, self.enemy_state_feat_dim, -1)

        padded_belief = th.zeros(bs, self.n_agents, self.args.state_shape, device=ep_batch.device)
        flat_enemy_belief = self.belief_mu.reshape(bs, self.n_agents, -1)
        padded_belief[:, :, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim] = flat_enemy_belief
        ep_batch["belief"][:, t] = padded_belief

        return agent_outs.view(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.history_tokens = None
        self.belief_mu = None
        self.belief_logvar = None
        self.belief_u = None

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
