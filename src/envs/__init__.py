from functools import partial
import os
from smac.env import MultiAgentEnv
from .starcraft import StarCraft2Env
from .stag_hunt import StagHunt
from .foraging import ForagingEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)

os.environ.setdefault("SC2PATH", "/home/liwenlei/StarCraftII")
