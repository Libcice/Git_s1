import torch
import torch.nn.functional as F

from mappo_onpolicy.models.util import check
from mappo_belief.smac_layout import build_smac_token_layout


class SMACTokenizer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.layout = build_smac_token_layout(args)
        self.obs_last_action = bool(getattr(args, "obs_last_action", False))
        self.obs_agent_id = bool(getattr(args, "obs_agent_id", False))
        self.tpdv = dict(dtype=torch.float32, device=device)

    def _pad_tokens(self, tensor):
        pad = self.layout.token_dim - tensor.size(-1)
        if pad <= 0:
            return tensor
        return F.pad(tensor, (0, pad))

    def _default_agent_ids(self, batch_size, device):
        if batch_size % self.args.n_agents != 0:
            raise ValueError("Cannot infer agent ids for tokenizer batch of size {}".format(batch_size))
        repeats = batch_size // self.args.n_agents
        return torch.eye(self.args.n_agents, device=device).repeat(repeats, 1)

    def __call__(self, obs, prev_actions=None, agent_ids=None):
        obs = check(obs).to(**self.tpdv)
        if prev_actions is not None:
            prev_actions = check(prev_actions).to(**self.tpdv)
        if agent_ids is not None:
            agent_ids = check(agent_ids).to(**self.tpdv)

        batch_size = obs.shape[0]
        device = obs.device
        cursor = 0

        move = obs[:, cursor:cursor + self.layout.move_dim]
        cursor += self.layout.move_dim

        enemy_total = self.args.enemy_num * self.layout.enemy_obs_feat_dim
        enemy = obs[:, cursor:cursor + enemy_total].reshape(
            batch_size, self.args.enemy_num, self.layout.enemy_obs_feat_dim
        )
        cursor += enemy_total

        ally_total = (self.args.n_agents - 1) * self.layout.ally_obs_feat_dim
        ally = obs[:, cursor:cursor + ally_total].reshape(
            batch_size, self.args.n_agents - 1, self.layout.ally_obs_feat_dim
        )
        cursor += ally_total

        own = obs[:, cursor:]
        extras = [own]
        if self.obs_last_action:
            if prev_actions is None:
                prev_actions = torch.zeros(batch_size, self.args.n_actions, device=device)
            extras.append(prev_actions)
        if self.obs_agent_id:
            if agent_ids is None:
                agent_ids = self._default_agent_ids(batch_size, device)
            extras.append(agent_ids)
        self_token = torch.cat(extras, dim=-1)

        move_token = self._pad_tokens(move).unsqueeze(1)
        self_token_padded = self._pad_tokens(self_token).unsqueeze(1)
        enemy_tokens = self._pad_tokens(enemy)
        ally_tokens = self._pad_tokens(ally)
        step_tokens = torch.cat([move_token, self_token_padded, enemy_tokens, ally_tokens], dim=1)

        current = {
            "move_token": self._pad_tokens(move),
            "self_token": self._pad_tokens(self_token),
            "enemy_tokens": self._pad_tokens(enemy),
            "enemy_visible": (enemy.abs().sum(dim=-1) > 0).float(),
            "ally_tokens": self._pad_tokens(ally),
            "ally_visible": (ally.abs().sum(dim=-1) > 0).float(),
        }
        return step_tokens, current
