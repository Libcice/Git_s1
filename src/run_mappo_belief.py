import datetime
import os
import pprint

import torch as th
from os.path import abspath, dirname
from types import SimpleNamespace as SN

from mappo_belief.runner import MAPPOBeliefRunner
from utils.logging import Logger


def run(_run, _config, _log):
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"
    args.num_env_steps = getattr(args, "num_env_steps", getattr(args, "t_max", 0))
    args.eval_interval = getattr(args, "eval_interval", getattr(args, "test_interval", 10000))
    args.eval_episodes = getattr(args, "eval_episodes", getattr(args, "test_nepisode", 32))
    args.n_eval_rollout_threads = getattr(args, "n_eval_rollout_threads", 1)
    args.use_eval = getattr(args, "use_eval", True)
    args.use_linear_lr_decay = getattr(args, "use_linear_lr_decay", False)

    logger = Logger(_log)
    logger.setup_sacred(_run)

    map_name = args.env_args.get("map_name", args.env)
    algo_name = args.name
    run_id = str(getattr(_run, "_id", "adhoc"))
    unique_token = "{}__{}__{}".format(
        args.name,
        map_name,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    results_root = os.path.join(dirname(dirname(abspath(__file__))), args.local_results_path)
    run_dir = os.path.join(results_root, "sacred", str(map_name), str(algo_name), run_id)
    os.makedirs(run_dir, exist_ok=True)

    if args.use_tensorboard:
        tb_root = os.path.join(results_root, "tb_logs", str(map_name), str(algo_name))
        os.makedirs(tb_root, exist_ok=True)
        logger.setup_tb(os.path.join(tb_root, unique_token))

    logger.console_logger.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    logger.console_logger.info("\n\n%s\n", experiment_params)

    runner = MAPPOBeliefRunner(args, logger, run_dir)
    try:
        runner.run()
    finally:
        runner.close()
