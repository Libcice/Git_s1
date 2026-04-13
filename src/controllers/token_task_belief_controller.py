from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class TokenTaskBeliefMAC:
    """MAC for token-transformer agents with persistent task-belief slots."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self._setup_layout(scheme)
        self._build_agents(self.token_dim)

        self.hidden_states = None
        self.belief_slots = None
        self.slot_latent = None
        self.belief_attn = None
        self.belief_context = None
        self.q_context = None
        self.enemy_ctx = None
        self.unseen_steps = None
        self.prev_enemy_visible = None

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

    def _write_belief_summary(self, ep_batch, t):
        if getattr(self.args, "record_belief_summary", True) is False:
            return
        if not hasattr(ep_batch, "scheme") or "belief" not in ep_batch.scheme:
            return
        belief_summary = th.zeros(
            ep_batch.batch_size,
            self.n_agents,
            self.args.state_shape,
            device=ep_batch.device,
        )
        ep_batch["belief"][:, t] = belief_summary

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def select_actions_vis(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions, self.hidden_states, agent_outputs

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        current = self._build_step_inputs(ep_batch, t)
        current["unseen_steps"] = self.unseen_steps.reshape(bs * self.n_agents, self.args.enemy_num)
        current["prev_enemy_visible"] = self.prev_enemy_visible.reshape(bs * self.n_agents, self.args.enemy_num)
        hidden = self.hidden_states.reshape(bs * self.n_agents, -1)
        belief_slots = self.belief_slots.reshape(bs * self.n_agents, self.args.enemy_num, -1)
        (
            agent_outs,
            new_hidden,
            new_belief_slots,
            slot_latent,
            belief_attn,
            belief_context,
            q_context,
            enemy_ctx,
        ) = self.agent(current, hidden, belief_slots)

        self.hidden_states = new_hidden.reshape(bs, self.n_agents, -1)
        self.belief_slots = new_belief_slots.reshape(bs, self.n_agents, self.args.enemy_num, -1)
        self.slot_latent = slot_latent.reshape(bs, self.n_agents, self.args.enemy_num, -1)
        self.belief_attn = belief_attn.reshape(bs, self.n_agents, self.args.enemy_num)
        self.belief_context = belief_context.reshape(bs, self.n_agents, -1)
        self.q_context = q_context.reshape(bs, self.n_agents, -1)
        self.enemy_ctx = enemy_ctx.reshape(bs, self.n_agents, self.args.enemy_num, -1)
        self._write_belief_summary(ep_batch, t)

        current_visible = current["enemy_visible"].reshape(bs, self.n_agents, self.args.enemy_num)
        self.unseen_steps = th.where(
            current_visible > 0,
            th.zeros_like(self.unseen_steps),
            self.unseen_steps + 1.0,
        )
        self.prev_enemy_visible = current_visible

        return agent_outs.reshape(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        device = self.agent.init_hidden().device
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).clone()
        self.belief_slots = self.agent.init_belief_slots().unsqueeze(0).expand(
            batch_size,
            self.n_agents,
            -1,
            -1,
        ).clone()
        self.slot_latent = None
        self.belief_attn = None
        self.belief_context = None
        self.q_context = None
        self.enemy_ctx = None
        self.unseen_steps = th.zeros(batch_size, self.n_agents, self.args.enemy_num, device=device)
        self.prev_enemy_visible = th.zeros(batch_size, self.n_agents, self.args.enemy_num, device=device)

    def get_belief(self):
        return self.belief_slots

    def get_slot_latent(self):
        return self.slot_latent

    def get_belief_attn(self):
        return self.belief_attn

    def get_belief_context(self):
        return self.belief_context

    def get_q_context(self):
        return self.q_context

    def get_enemy_context(self):
        return self.enemy_ctx

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def share(self):
        pass
