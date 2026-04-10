import torch

from mappo_belief.models.actor_critic import BeliefCritic, BeliefTransformerActor, ResidualBeliefActor
from mappo_onpolicy.utils.util import update_linear_schedule


class R_MAPPOBeliefPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.use_belief_critic = getattr(args, "use_belief_critic", False)
        self.use_residual_belief_actor = getattr(args, "use_residual_belief_actor", False)

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        if self.use_residual_belief_actor:
            self.actor = ResidualBeliefActor(args, self.obs_space, self.act_space, self.device)
        else:
            self.actor = BeliefTransformerActor(args, self.obs_space, self.act_space, self.device)
        self.critic = BeliefCritic(args, self.share_obs_space, self.device)
        self.actor_state_size = getattr(self.actor, "actor_state_size", args.hidden_size)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def _extract_critic_context(self, aux):
        if not self.use_belief_critic:
            return None
        if aux is None or "critic_context" not in aux:
            raise ValueError("critic_context is required when use_belief_critic=True")
        return aux["critic_context"].detach()

    def _encode_critic_context(self, obs, prev_actions, agent_ids, rnn_states_actor, masks):
        if not self.use_belief_critic:
            return None
        return self.actor.get_critic_context(
            obs,
            prev_actions,
            agent_ids,
            rnn_states_actor,
            masks,
        ).detach()

    def get_actions(
        self,
        cent_obs,
        obs,
        prev_actions,
        agent_ids,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        return_aux=False,
    ):
        need_aux = return_aux or self.use_belief_critic
        actor_out = self.actor(
            obs,
            prev_actions,
            agent_ids,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
            return_aux=need_aux,
        )
        if need_aux:
            actions, action_log_probs, rnn_states_actor, aux = actor_out
        else:
            actions, action_log_probs, rnn_states_actor = actor_out
            aux = None

        critic_context = self._extract_critic_context(aux)
        values, rnn_states_critic = self.critic(cent_obs, critic_context, rnn_states_critic, masks)
        if return_aux:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, aux
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(
        self,
        cent_obs,
        obs,
        prev_actions,
        agent_ids,
        rnn_states_actor,
        rnn_states_critic,
        masks,
    ):
        critic_context = self._encode_critic_context(
            obs,
            prev_actions,
            agent_ids,
            rnn_states_actor,
            masks,
        )
        values, _ = self.critic(cent_obs, critic_context, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        cent_obs,
        obs,
        prev_actions,
        agent_ids,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        return_aux=False,
    ):
        need_aux = return_aux or self.use_belief_critic
        actor_out = self.actor.evaluate_actions(
            obs,
            prev_actions,
            agent_ids,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
            return_aux=need_aux,
        )
        if need_aux:
            action_log_probs, dist_entropy, aux = actor_out
        else:
            action_log_probs, dist_entropy = actor_out
            aux = None

        critic_context = self._extract_critic_context(aux)
        values, _ = self.critic(cent_obs, critic_context, rnn_states_critic, masks)
        if return_aux:
            return values, action_log_probs, dist_entropy, aux
        return values, action_log_probs, dist_entropy

    def act(
        self,
        obs,
        prev_actions,
        agent_ids,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        actions, _, rnn_states_actor = self.actor(
            obs,
            prev_actions,
            agent_ids,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states_actor


MAPPOBeliefPolicy = R_MAPPOBeliefPolicy
