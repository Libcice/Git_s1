import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

# 地图参数字典 - 在 main.py 中定义，供 run.py 导入使用
MAP_DICT = {
    "3m":{"ally_num":3,"enemy_num":3},
    "8m":{"ally_num":8,"enemy_num":8},
    "2s3z":{"ally_num":5,"enemy_num":5},
    "3s5z":{"ally_num":8,"enemy_num":8},
    "5m_vs_6m":{"ally_num":5,"enemy_num":6},
    "8m_vs_9m":{"ally_num":8,"enemy_num":9},
    "10m_vs_11m":{"ally_num":10,"enemy_num":11},
    "3s5z_vs_3s6z":{"ally_num":8,"enemy_num":9},
    "2m_vs_1z":{"ally_num":2,"enemy_num":1},
    "2s_vs_1sc":{"ally_num":2,"enemy_num":1},
    "3s_vs_3z":{"ally_num":3,"enemy_num":3},
    "3s_vs_4z":{"ally_num":3,"enemy_num":4},
    "3s_vs_5z":{"ally_num":3,"enemy_num":5},
    "6h_vs_8z":{"ally_num":6,"enemy_num":8},
    "corridor":{"ally_num":6,"enemy_num":24},
    "2c_vs_64zg":{"ally_num":2,"enemy_num":64},
    "1c3s5z":{"ally_num":9,"enemy_num":9},
    "MMM":{"ally_num":10,"enemy_num":10},
    "MMM2":{"ally_num":10,"enemy_num":12},
    "7sz":{"ally_num":14,"enemy_num":14},
    "5s10z":{"ally_num":15,"enemy_num":15},
    "1c3s5z_vs_1c3s6z":{"ally_num":9,"enemy_num":10},
    "1c3s8z_vs_1c3s9z":{"ally_num":12,"enemy_num":13},
    "pp-2":{"ally_num":8,"enemy_num":8},
    "pp-1":{"ally_num":8,"enemy_num":8},
    "pp-0.5":{"ally_num":8,"enemy_num":8},
    "lbf-4-2":{"ally_num":4,"enemy_num":4},
    "lbf-4-4":{"ally_num":4,"enemy_num":4},
    "lbf-3-3":{"ally_num":3,"enemy_num":3},
}

from run import run

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    
    # 根据地图名称设置 ally_num 和 enemy_num（在 Sacred 记录之前）
    try:
        map_name = config["env_args"]["map_name"]
    except:
        map_name = config["env_args"]["key"]
    
    if map_name in MAP_DICT:
        config["ally_num"] = MAP_DICT[map_name]["ally_num"]
        config["enemy_num"] = MAP_DICT[map_name]["enemy_num"]
        _log.info(f"Map {map_name}: ally_num={config['ally_num']}, enemy_num={config['enemy_num']}")
    else:
        _log.warning(f"Map {map_name} not in MAP_DICT, using default ally_num={config.get('ally_num', 'N/A')}, enemy_num={config.get('enemy_num', 'N/A')}")
    
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)
    gpu_id="0"
    try:
        map_name = config_dict["gpu_id"]
        gpu_id = str(parse_command(params, "gpu_id", gpu_id))
    except:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name'])
    file_obs_path = join(results_path, "sacred", map_name, algo_name)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

