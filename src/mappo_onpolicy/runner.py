import os
import time

import numpy as np
import torch

from mappo_onpolicy.buffer import SharedReplayBuffer
from mappo_onpolicy.env import ParallelSMACEnv
from mappo_onpolicy.learner import R_MAPPO
from mappo_onpolicy.policy import R_MAPPOPolicy
from utils.timehelper import time_left, time_str


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACOnPolicyRunner:
    def __init__(self, args, logger, run_dir):
        self.args = args
        self.logger = logger
        self.run_dir = run_dir
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        self.n_rollout_threads = args.batch_size_run
        self.n_eval_rollout_threads = getattr(args, "n_eval_rollout_threads", 1)
        self.eval_episodes = getattr(args, "eval_episodes", getattr(args, "test_nepisode", 32))
        self.eval_interval = getattr(args, "eval_interval", getattr(args, "test_interval", 10000))
        self.log_interval = getattr(args, "log_interval", 10000)
        self.save_interval = getattr(args, "save_model_interval", 2000000)
        self.num_env_steps = getattr(args, "num_env_steps", getattr(args, "t_max", 0))
        self.use_eval = getattr(args, "use_eval", True)
        self.use_linear_lr_decay = getattr(args, "use_linear_lr_decay", False)
        self.model_dir = os.path.join(run_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.envs = ParallelSMACEnv(args, self.n_rollout_threads, seed_offset=0)
        self.eval_envs = ParallelSMACEnv(args, self.n_eval_rollout_threads, seed_offset=100000) if self.use_eval else None

        env_info = self.envs.env_info
        self.num_agents = env_info["n_agents"]
        self.episode_length = env_info["episode_limit"]
        self.args.episode_length = self.episode_length
        self.args.n_rollout_threads = self.n_rollout_threads
        self.args.n_eval_rollout_threads = self.n_eval_rollout_threads

        share_obs_space = (
            self.envs.share_observation_space[0]
            if self.args.use_centralized_V
            else self.envs.observation_space[0]
        )

        self.policy = R_MAPPOPolicy(
            self.args,
            self.envs.observation_space[0],
            share_obs_space,
            self.envs.action_space[0],
            device=self.device,
        )
        self.trainer = R_MAPPO(self.args, self.policy, device=self.device)
        self.buffer = SharedReplayBuffer(
            self.args,
            self.num_agents,
            self.envs.observation_space[0],
            share_obs_space,
            self.envs.action_space[0],
        )

        self.total_num_steps = 0
        self.train_returns = []
        self.train_wins = []
        self.train_lengths = []
        self.train_episode_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)
        self.train_episode_lengths = np.zeros(self.n_rollout_threads, dtype=np.int32)
        self.completed_train_episodes = 0

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        if not self.args.use_centralized_V:
            share_obs = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = self.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        if np.any(dones_env):
            rnn_states[dones_env] = np.zeros(
                ((dones_env == True).sum(), self.num_agents, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32,
            )
            rnn_states_critic[dones_env] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.args.recurrent_N,
                    self.args.hidden_size,
                ),
                dtype=np.float32,
            )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env] = 0.0

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones] = 0.0
        active_masks[dones_env] = 1.0

        bad_masks = np.array(
            [
                [[0.0] if infos[env_i][agent_i].get("bad_transition", False) else [1.0] for agent_i in range(self.num_agents)]
                for env_i in range(self.n_rollout_threads)
            ],
            dtype=np.float32,
        )

        if not self.args.use_centralized_V:
            share_obs = obs.copy()

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
        )

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    def save(self, total_num_steps):
        torch.save(self.policy.actor.state_dict(), os.path.join(self.model_dir, f"actor_{total_num_steps}.pt"))
        torch.save(self.policy.critic.state_dict(), os.path.join(self.model_dir, f"critic_{total_num_steps}.pt"))

    def run(self):
        self.warmup()
        start = time.time()
        last_time = start
        episodes = max(1, self.num_env_steps // self.episode_length // self.n_rollout_threads)
        last_log = 0
        last_eval = 0
        last_save = 0

        self.logger.console_logger.info("Beginning training for %s timesteps", self.num_env_steps)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                self.train_episode_returns += rewards[:, 0, 0]
                self.train_episode_lengths += 1
                dones_env = np.all(dones, axis=1)
                for env_i, done in enumerate(dones_env):
                    if done:
                        self.train_returns.append(float(self.train_episode_returns[env_i]))
                        self.train_lengths.append(int(self.train_episode_lengths[env_i]))
                        self.train_wins.append(float(infos[env_i][0].get("won", False)))
                        self.train_episode_returns[env_i] = 0.0
                        self.train_episode_lengths[env_i] = 0
                        self.completed_train_episodes += 1

                self.insert(
                    (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                )
                self.total_num_steps += self.n_rollout_threads

            self.compute()
            train_info = self.train()

            if self.total_num_steps - last_log >= self.log_interval:
                self.logger.log_stat("episode", self.completed_train_episodes, self.total_num_steps)
                if self.train_returns:
                    self.logger.log_stat("return_mean", np.mean(self.train_returns), self.total_num_steps)
                    self.logger.log_stat("return_std", np.std(self.train_returns), self.total_num_steps)
                    self.logger.log_stat("battle_won_mean", np.mean(self.train_wins), self.total_num_steps)
                    self.logger.log_stat("ep_length_mean", np.mean(self.train_lengths), self.total_num_steps)
                    self.train_returns.clear()
                    self.train_wins.clear()
                    self.train_lengths.clear()

                self.logger.log_stat("mappo_value_loss", train_info["value_loss"], self.total_num_steps)
                self.logger.log_stat("mappo_policy_loss", train_info["policy_loss"], self.total_num_steps)
                self.logger.log_stat("mappo_entropy", train_info["dist_entropy"], self.total_num_steps)
                self.logger.log_stat("mappo_actor_grad_norm", float(train_info["actor_grad_norm"]), self.total_num_steps)
                self.logger.log_stat("mappo_critic_grad_norm", float(train_info["critic_grad_norm"]), self.total_num_steps)
                self.logger.log_stat("mappo_ratio", float(train_info["ratio"]), self.total_num_steps)
                self.logger.console_logger.info("t_env: %s / %s", self.total_num_steps, self.num_env_steps)
                self.logger.console_logger.info(
                    "Estimated time left: %s. Time passed: %s",
                    time_left(last_time, last_log, self.total_num_steps, self.num_env_steps),
                    time_str(time.time() - start),
                )
                last_time = time.time()
                self.logger.print_recent_stats()
                last_log = self.total_num_steps

            if self.use_eval and self.total_num_steps - last_eval >= self.eval_interval:
                self.eval(self.total_num_steps)
                last_eval = self.total_num_steps

            if self.args.save_model and self.total_num_steps - last_save >= self.save_interval:
                self.save(self.total_num_steps)
                last_save = self.total_num_steps

        if self.use_eval:
            self.eval(self.total_num_steps)
        self.logger.console_logger.info("Finished Training")

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_returns = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
        completed_returns = []
        completed_wins = []
        completed = 0

        while completed < self.eval_episodes:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions
            )
            eval_returns += eval_rewards[:, 0, 0]
            eval_dones_env = np.all(eval_dones, axis=1)

            if np.any(eval_dones_env):
                eval_rnn_states[eval_dones_env] = 0.0
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env] = 0.0

            for env_i, done in enumerate(eval_dones_env):
                if done:
                    completed += 1
                    completed_returns.append(float(eval_returns[env_i]))
                    completed_wins.append(float(eval_infos[env_i][0].get("won", False)))
                    eval_returns[env_i] = 0.0
                    if completed >= self.eval_episodes:
                        break

        self.logger.log_stat("test_return_mean", np.mean(completed_returns), total_num_steps)
        self.logger.log_stat("test_return_std", np.std(completed_returns), total_num_steps)
        self.logger.log_stat("test_battle_won_mean", np.mean(completed_wins), total_num_steps)

    def close(self):
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
