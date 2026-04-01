import numpy as np

from mappo_onpolicy.buffer import SharedReplayBuffer as BaseSharedReplayBuffer, _cast, _flatten


class BeliefReplayBuffer(BaseSharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)
        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("mappo_belief currently expects discrete action space")
        self.actor_state_size = getattr(args, "actor_state_size", self.hidden_size)
        if self.actor_state_size != self.hidden_size:
            self.rnn_states = np.zeros(
                (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.actor_state_size),
                dtype=np.float32,
            )

        self.prev_actions = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
            dtype=np.float32,
        )
        self.agent_ids = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, num_agents),
            dtype=np.float32,
        )
        eye = np.eye(num_agents, dtype=np.float32).reshape(1, 1, num_agents, num_agents)
        self.agent_ids[:] = eye

    def insert(
        self,
        share_obs,
        obs,
        prev_actions,
        agent_ids,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        next_idx = self.step + 1
        super().insert(
            share_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks=bad_masks,
            active_masks=active_masks,
            available_actions=available_actions,
        )
        self.prev_actions[next_idx] = prev_actions.copy()
        self.agent_ids[next_idx] = agent_ids.copy()

    def after_update(self):
        super().after_update()
        self.prev_actions[0] = self.prev_actions[-1].copy()
        self.agent_ids[0] = self.agent_ids[-1].copy()

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        rand = np.random.permutation(data_chunks)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = _cast(self.share_obs[:-1])
        obs = _cast(self.obs[:-1])
        prev_actions = _cast(self.prev_actions[:-1])
        agent_ids = _cast(self.agent_ids[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )
        available_actions = _cast(self.available_actions[:-1]) if self.available_actions is not None else None

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            prev_actions_batch = []
            agent_ids_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                prev_actions_batch.append(prev_actions[ind:ind + data_chunk_length])
                agent_ids_batch.append(agent_ids[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            prev_actions_batch = np.stack(prev_actions_batch, axis=1)
            agent_ids_batch = np.stack(agent_ids_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            prev_actions_batch = _flatten(L, N, prev_actions_batch)
            agent_ids_batch = _flatten(L, N, agent_ids_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield (
                share_obs_batch,
                obs_batch,
                prev_actions_batch,
                agent_ids_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )


SharedReplayBuffer = BeliefReplayBuffer
