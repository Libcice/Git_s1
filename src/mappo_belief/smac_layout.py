from dataclasses import dataclass


@dataclass
class SMACTokenLayout:
    unit_type_bits: int
    move_dim: int
    enemy_obs_feat_dim: int
    ally_obs_feat_dim: int
    raw_own_dim: int
    self_token_raw_dim: int
    token_dim: int
    tokens_per_step: int
    enemy_state_feat_dim: int
    ally_state_feat_dim: int
    ally_state_dim: int
    enemy_state_dim: int


def build_smac_token_layout(args):
    env_args = getattr(args, "env_args", {})
    n_agents = int(args.n_agents)
    enemy_num = int(args.enemy_num)
    obs_shape = int(args.obs_shape)
    n_actions = int(args.n_actions)
    obs_last_action = bool(getattr(args, "obs_last_action", False))
    obs_agent_id = bool(getattr(args, "obs_agent_id", False))

    unit_type_bits = max(int(getattr(args, "unit_dim", 5)) - 5, 0)
    move_dim = 4
    if env_args.get("obs_pathing_grid", False):
        move_dim += 8
    if env_args.get("obs_terrain_height", False):
        move_dim += 9

    health_bits = 2 if env_args.get("obs_all_health", True) else 0
    enemy_obs_feat_dim = 4 + unit_type_bits + health_bits
    ally_obs_feat_dim = 4 + unit_type_bits + health_bits
    if env_args.get("obs_last_action", False):
        ally_obs_feat_dim += n_actions

    raw_own_dim = obs_shape - move_dim - enemy_num * enemy_obs_feat_dim - (n_agents - 1) * ally_obs_feat_dim
    self_token_raw_dim = raw_own_dim
    if obs_last_action:
        self_token_raw_dim += n_actions
    if obs_agent_id:
        self_token_raw_dim += n_agents

    token_dim = max(move_dim, enemy_obs_feat_dim, ally_obs_feat_dim, self_token_raw_dim)
    enemy_state_feat_dim = 4 + unit_type_bits
    ally_state_feat_dim = 5 + unit_type_bits
    ally_state_dim = n_agents * ally_state_feat_dim
    enemy_state_dim = enemy_num * enemy_state_feat_dim

    return SMACTokenLayout(
        unit_type_bits=unit_type_bits,
        move_dim=move_dim,
        enemy_obs_feat_dim=enemy_obs_feat_dim,
        ally_obs_feat_dim=ally_obs_feat_dim,
        raw_own_dim=raw_own_dim,
        self_token_raw_dim=self_token_raw_dim,
        token_dim=token_dim,
        tokens_per_step=2 + enemy_num + (n_agents - 1),
        enemy_state_feat_dim=enemy_state_feat_dim,
        ally_state_feat_dim=ally_state_feat_dim,
        ally_state_dim=ally_state_dim,
        enemy_state_dim=enemy_state_dim,
    )
