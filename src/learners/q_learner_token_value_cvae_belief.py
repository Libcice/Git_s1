import copy
import math
import os

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerTokenValueCVAEBelief:
    """QMIX learner for token-history CVAE belief with value-aware distillation."""

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.online_params = list(self.mac_params)
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.online_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.unit_type_bits = max(int(getattr(self.args, "unit_dim", 5)) - 5, 0)
        self.enemy_state_feat_dim = getattr(self.args, "enemy_state_feat_dim", 4 + self.unit_type_bits)
        self.enemy_state_dim = getattr(self.args, "enemy_state_dim", self.args.enemy_num * self.enemy_state_feat_dim)
        self.ally_state_feat_dim = 5 + self.unit_type_bits
        self.ally_state_dim = self.args.n_agents * self.ally_state_feat_dim
        env_args = getattr(self.args, "env_args", {})
        self.move_dim = 4
        if env_args.get("obs_pathing_grid", False):
            self.move_dim += 8
        if env_args.get("obs_terrain_height", False):
            self.move_dim += 9
        health_bits = 1 + 1 if env_args.get("obs_all_health", True) else 0
        self.enemy_obs_feat_dim = 4 + self.unit_type_bits + health_bits

        self.hidden_dim = getattr(self.args, "transformer_hidden_dim", self.args.rnn_hidden_dim)
        self.belief_loss_coef = getattr(self.args, "belief_loss_coef", 0.05)
        self.belief_posterior_coef = getattr(self.args, "belief_posterior_coef", 0.25)
        self.belief_kl_coef = getattr(self.args, "belief_kl_coef", 0.1)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -3.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_nll_clip = getattr(self.args, "belief_nll_clip", 2.0)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 300000)
        self.belief_kl_warmup_t = getattr(self.args, "belief_kl_warmup_t", self.belief_warmup_t)
        self.belief_pretrain_t = getattr(self.args, "belief_pretrain_t", 50000)
        self.belief_pretrain_only = getattr(self.args, "belief_pretrain_only", False)

        self.belief_repr_coef = getattr(self.args, "belief_repr_coef", 0.01)
        self.belief_aux_delta_q_coef = getattr(self.args, "belief_aux_delta_q_coef", 0.015)
        self.belief_distill_importance_min = getattr(self.args, "belief_distill_importance_min", 0.25)
        self.belief_distill_importance_max = getattr(self.args, "belief_distill_importance_max", 4.0)
        self.belief_value_warmup_t = getattr(self.args, "belief_value_warmup_t", 600000)

        self.disable_belief = getattr(self.args, "disable_belief", False)
        self.use_belief_for_q = getattr(self.args, "use_belief_for_q", True) and not self.disable_belief
        self.belief_prior_type = getattr(self.args, "belief_prior_type", "conditional")
        if self.disable_belief:
            self.belief_loss_coef = 0.0
            self.belief_kl_coef = 0.0
            self.belief_repr_coef = 0.0
            self.belief_aux_delta_q_coef = 0.0
            self.belief_pretrain_only = False

        self.posterior_enabled = self.belief_loss_coef > 0.0
        self.value_losses_enabled = self.belief_repr_coef > 0.0 or self.belief_aux_delta_q_coef > 0.0
        self.belief_enabled = self.posterior_enabled or self.use_belief_for_q or self.value_losses_enabled
        self.q_learning_started = False

        self.optimiser = RMSprop(params=self.online_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _smooth_ramp(self, progress):
        progress = max(0.0, min(1.0, float(progress)))
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def _stabilized_gaussian_nll(self, target, mu, logvar):
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        inv_var = th.exp(-logvar)
        per_dim_nll = 0.5 * (((target - mu) ** 2) * inv_var + logvar + math.log(2.0 * math.pi))
        raw_nll = per_dim_nll.mean(dim=-1)
        nll_floor = 0.5 * (self.belief_logvar_min + math.log(2.0 * math.pi))
        stable_nll = (raw_nll - nll_floor).clamp(min=0.0, max=self.belief_nll_clip)
        return raw_nll, stable_nll

    def _kl_divergence(self, post_mu, post_logvar, prior_mu, prior_logvar):
        post_logvar = post_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        prior_logvar = prior_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)

        if self.belief_prior_type == "standard_normal":
            kl = 0.5 * (th.exp(post_logvar) + post_mu.pow(2) - 1.0 - post_logvar).sum(dim=-1)
        else:
            diff = post_mu - prior_mu
            kl = 0.5 * (
                prior_logvar - post_logvar + (th.exp(post_logvar) + diff.pow(2)) * th.exp(-prior_logvar) - 1.0
            ).sum(dim=-1)

        latent_dim = float(max(1, getattr(self.args, "belief_latent_dim", 1)))
        return kl / latent_dim

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _target_q_from_enemy_summary(self, q_context_t, enemy_summary_t, allow_input_grad=False):
        bs, n_agents, _ = q_context_t.shape
        target_q_head = self.target_mac.agent.q_head
        q_input_t = th.cat([q_context_t.detach(), enemy_summary_t], dim=-1).reshape(bs * n_agents, self.hidden_dim * 2)
        linear_kwargs = {
            "input": q_input_t,
            "weight": target_q_head.weight.detach(),
            "bias": None if target_q_head.bias is None else target_q_head.bias.detach(),
        }
        if allow_input_grad:
            q_values_t = F.linear(**linear_kwargs)
        else:
            with th.no_grad():
                q_values_t = F.linear(**linear_kwargs)
        return q_values_t.view(bs, n_agents, self.args.n_actions)

    def _gather_action_values(self, q_values_t, actions_t):
        return th.gather(q_values_t, dim=-1, index=actions_t).squeeze(-1)

    def _build_oracle_targets(
        self,
        q_context_t,
        visible_enemy_summary_t,
        hidden_enemy_view,
        enemy_visible_t,
        student_hidden_summary_t=None,
    ):
        bs, n_agents, n_enemies, _ = hidden_enemy_view.shape
        target_agent = self.target_mac.agent
        with th.no_grad():
            oracle_hidden_feat_t = F.relu(
                target_agent.belief_enemy_proj(
                    hidden_enemy_view.reshape(bs * n_agents * n_enemies, self.enemy_state_feat_dim)
                )
            ).view(bs * n_agents, n_enemies, self.hidden_dim)
            hidden_enemy_mask_t = (1.0 - enemy_visible_t).reshape(bs * n_agents, n_enemies).float()
            oracle_hidden_summary_t = target_agent._cross_attention_readout(
                q_context_t.reshape(bs * n_agents, self.hidden_dim),
                oracle_hidden_feat_t,
                hidden_enemy_mask_t,
                target_agent.hidden_summary_query_proj,
                target_agent.hidden_summary_key_proj,
                target_agent.hidden_summary_value_proj,
            ).view(bs, n_agents, self.hidden_dim)
        q_visible_t = self._target_q_from_enemy_summary(q_context_t, visible_enemy_summary_t.detach())
        q_student_t = None
        if student_hidden_summary_t is not None:
            q_student_t = self._target_q_from_enemy_summary(
                q_context_t,
                visible_enemy_summary_t.detach() + student_hidden_summary_t,
                allow_input_grad=True,
            )
        q_oracle_t = self._target_q_from_enemy_summary(
            q_context_t,
            visible_enemy_summary_t.detach() + oracle_hidden_summary_t.detach(),
        )
        delta_q_target_t = q_oracle_t - q_visible_t
        student_delta_q_t = None if q_student_t is None else (q_student_t - q_visible_t)
        return oracle_hidden_summary_t.detach(), delta_q_target_t.detach(), student_delta_q_t, q_visible_t.detach(), q_student_t, q_oracle_t.detach()

    def _belief_action_diagnostics(self, q_visible_t, q_student_t, q_oracle_t):
        with th.no_grad():
            a_vis_t = q_visible_t.argmax(dim=-1)
            a_bel_t = q_student_t.argmax(dim=-1)
            a_oracle_t = q_oracle_t.argmax(dim=-1)
            help_t = ((a_vis_t != a_oracle_t) & (a_bel_t == a_oracle_t)).float()
            drag_t = ((a_vis_t == a_oracle_t) & (a_bel_t != a_oracle_t)).float()
            net_gain_t = help_t - drag_t
            student_delta_t = q_student_t - q_visible_t
            target_delta_t = q_oracle_t - q_visible_t
            delta_cosine_t = F.cosine_similarity(student_delta_t, target_delta_t, dim=-1)
        return help_t, drag_t, net_gain_t, delta_cosine_t

    def _build_distill_importance(self, delta_q_target_t, distill_mask_t):
        with th.no_grad():
            if delta_q_target_t.dim() > distill_mask_t.dim():
                raw_importance_t = delta_q_target_t.abs().mean(dim=-1)
            else:
                raw_importance_t = delta_q_target_t.abs()
            agent_count_t = distill_mask_t.sum(dim=-1, keepdim=True)
            mean_importance_t = (raw_importance_t * distill_mask_t).sum(dim=-1, keepdim=True) / agent_count_t.clamp(min=1.0)
            normalized_importance_t = raw_importance_t / mean_importance_t.clamp(min=1e-6)
            normalized_importance_t = normalized_importance_t.clamp(
                min=self.belief_distill_importance_min,
                max=self.belief_distill_importance_max,
            )
            normalized_importance_t = normalized_importance_t * distill_mask_t
            normalized_importance_t = th.where(
                agent_count_t > 0,
                normalized_importance_t,
                th.zeros_like(normalized_importance_t),
            )
        return normalized_importance_t.detach()

    def _optimizer_state_subset(self, opt_state, start, count):
        if opt_state is None or count <= 0:
            return None

        param_groups = opt_state.get("param_groups", [])
        if len(param_groups) != 1:
            return None

        saved_params = list(param_groups[0].get("params", []))
        if len(saved_params) < start + count:
            return None

        subset_ids = saved_params[start:start + count]
        subset_state = copy.deepcopy(opt_state)
        subset_state["param_groups"] = [copy.deepcopy(param_groups[0])]
        subset_state["param_groups"][0]["params"] = subset_ids
        opt_states = opt_state.get("state", {})
        subset_state["state"] = {
            param_id: copy.deepcopy(opt_states[param_id])
            for param_id in subset_ids
            if param_id in opt_states
        }
        return subset_state

    def _load_optimizer_state(self, optimiser, opt_state):
        if opt_state is None:
            return False
        try:
            optimiser.load_state_dict(opt_state)
            return True
        except ValueError:
            return False

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        belief_pretraining_active = self.belief_pretrain_only and (t_env < self.belief_pretrain_t)
        q_learning_active = (not self.belief_pretrain_only) or (t_env >= self.belief_pretrain_t)
        if q_learning_active and not self.q_learning_started:
            self._update_targets()
            self.q_learning_started = True

        mac_value_active = q_learning_active and self.value_losses_enabled
        hidden_enemy_state_active = self.posterior_enabled or belief_pretraining_active or mac_value_active

        mac_out = []
        belief_recon_sum = rewards.new_tensor(0.0)
        belief_recon_denom = rewards.new_tensor(0.0)
        belief_prior_nll_sum = rewards.new_tensor(0.0)
        belief_prior_nll_denom = rewards.new_tensor(0.0)
        belief_kl_sum = rewards.new_tensor(0.0)
        belief_kl_denom = rewards.new_tensor(0.0)
        prior_state_mu_sum = rewards.new_tensor(0.0)
        prior_state_mu_count = rewards.new_tensor(0.0)
        prior_state_logvar_sum = rewards.new_tensor(0.0)
        prior_state_logvar_count = rewards.new_tensor(0.0)
        post_state_mu_sum = rewards.new_tensor(0.0)
        post_state_mu_count = rewards.new_tensor(0.0)
        post_state_logvar_sum = rewards.new_tensor(0.0)
        post_state_logvar_count = rewards.new_tensor(0.0)
        prior_z_mu_sum = rewards.new_tensor(0.0)
        prior_z_mu_count = rewards.new_tensor(0.0)
        prior_z_logvar_sum = rewards.new_tensor(0.0)
        prior_z_logvar_count = rewards.new_tensor(0.0)
        post_z_mu_sum = rewards.new_tensor(0.0)
        post_z_mu_count = rewards.new_tensor(0.0)
        post_z_logvar_sum = rewards.new_tensor(0.0)
        post_z_logvar_count = rewards.new_tensor(0.0)
        prior_conf_sum = rewards.new_tensor(0.0)
        prior_conf_count = rewards.new_tensor(0.0)
        repr_loss_sum = rewards.new_tensor(0.0)
        repr_loss_denom = rewards.new_tensor(0.0)
        aux_delta_q_loss_sum = rewards.new_tensor(0.0)
        aux_delta_q_loss_denom = rewards.new_tensor(0.0)
        distill_importance_sum = rewards.new_tensor(0.0)
        distill_importance_count = rewards.new_tensor(0.0)
        oracle_delta_abs_sum = rewards.new_tensor(0.0)
        oracle_delta_abs_denom = rewards.new_tensor(0.0)
        prior_raw_nll_sum = rewards.new_tensor(0.0)
        prior_raw_nll_denom = rewards.new_tensor(0.0)
        post_raw_nll_sum = rewards.new_tensor(0.0)
        post_raw_nll_denom = rewards.new_tensor(0.0)
        belief_action_help_sum = rewards.new_tensor(0.0)
        belief_action_drag_sum = rewards.new_tensor(0.0)
        belief_action_net_gain_sum = rewards.new_tensor(0.0)
        belief_action_metric_denom = rewards.new_tensor(0.0)
        belief_delta_cosine_sum = rewards.new_tensor(0.0)
        belief_delta_cosine_denom = rewards.new_tensor(0.0)
        repr_weight = 0.0
        aux_delta_q_weight = 0.0
        value_warm = min(1.0, float(t_env) / float(max(1, self.belief_value_warmup_t)))

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim

        target_mac_out = None
        target_q_context_cache = None
        target_visible_enemy_summary_cache = None
        if q_learning_active:
            target_mac_out = []
            if mac_value_active:
                target_q_context_cache = []
                target_visible_enemy_summary_cache = []
            self.target_mac.init_hidden(batch.batch_size)
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
                    if mac_value_active and t < batch.max_seq_length - 1:
                        target_q_context_cache.append(self.target_mac.get_q_context().detach())
                        target_visible_enemy_summary_cache.append(self.target_mac.get_visible_enemy_summary().detach())

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            hidden_enemy_state_t = None
            if hidden_enemy_state_active:
                enemy_state_t = batch["state"][:, t, self.ally_state_dim:self.ally_state_dim + self.enemy_state_dim]
                enemy_state_t = enemy_state_t.view(batch.batch_size, self.args.enemy_num, self.enemy_state_feat_dim)
                hidden_enemy_state_t = enemy_state_t.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)
                hidden_enemy_state_t = hidden_enemy_state_t.reshape(
                    batch.batch_size * self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )

            agent_outs = self.mac.forward(
                batch,
                t=t,
                hidden_enemy_state=hidden_enemy_state_t if (self.posterior_enabled or belief_pretraining_active) else None,
            )
            mac_out.append(agent_outs)

            if not self.belief_enabled and not mac_value_active:
                continue

            prior_state_mu_t, prior_state_logvar_t = self.mac.get_prior_belief_stats()
            prior_z_mu_t, prior_z_logvar_t = self.mac.get_prior_latent_stats()
            prior_conf_t = self.mac.get_prior_belief_confidence()
            prior_state_mu_sum = prior_state_mu_sum + prior_state_mu_t.mean()
            prior_state_mu_count = prior_state_mu_count + prior_state_mu_t.new_tensor(1.0)
            prior_state_logvar_sum = prior_state_logvar_sum + prior_state_logvar_t.mean()
            prior_state_logvar_count = prior_state_logvar_count + prior_state_logvar_t.new_tensor(1.0)
            prior_z_mu_sum = prior_z_mu_sum + prior_z_mu_t.mean()
            prior_z_mu_count = prior_z_mu_count + prior_z_mu_t.new_tensor(1.0)
            prior_z_logvar_sum = prior_z_logvar_sum + prior_z_logvar_t.mean()
            prior_z_logvar_count = prior_z_logvar_count + prior_z_logvar_t.new_tensor(1.0)
            prior_conf_sum = prior_conf_sum + prior_conf_t.mean()
            prior_conf_count = prior_conf_count + prior_conf_t.new_tensor(1.0)

            hidden_enemy_view = None
            enemy_visible_t = None
            hidden_mask_t = None
            hidden_present_t = None
            if hidden_enemy_state_t is not None:
                enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
                enemy_obs_t = enemy_obs_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_obs_feat_dim,
                )
                enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
                hidden_enemy_view = hidden_enemy_state_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )
                alive_mask_t = (hidden_enemy_view[..., 0] > 0).float()
                time_mask_t = batch["filled"][:, t].float().unsqueeze(1).expand(-1, self.args.n_agents, -1)
                hidden_mask_t = (1.0 - enemy_visible_t) * alive_mask_t * time_mask_t
                hidden_present_t = (hidden_mask_t.sum(dim=-1) > 0).float()

            if hidden_enemy_state_t is not None and (self.posterior_enabled or belief_pretraining_active):
                posterior_state_mu_t, posterior_state_logvar_t = self.mac.get_posterior_belief_stats()
                posterior_z_mu_t, posterior_z_logvar_t = self.mac.get_posterior_latent_stats()
                post_state_mu_sum = post_state_mu_sum + posterior_state_mu_t.mean()
                post_state_mu_count = post_state_mu_count + posterior_state_mu_t.new_tensor(1.0)
                post_state_logvar_sum = post_state_logvar_sum + posterior_state_logvar_t.mean()
                post_state_logvar_count = post_state_logvar_count + posterior_state_logvar_t.new_tensor(1.0)
                post_z_mu_sum = post_z_mu_sum + posterior_z_mu_t.mean()
                post_z_mu_count = post_z_mu_count + posterior_z_mu_t.new_tensor(1.0)
                post_z_logvar_sum = post_z_logvar_sum + posterior_z_logvar_t.mean()
                post_z_logvar_count = post_z_logvar_count + posterior_z_logvar_t.new_tensor(1.0)

                posterior_state_mu_view = posterior_state_mu_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )
                posterior_state_logvar_view = posterior_state_logvar_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )
                prior_state_mu_view = prior_state_mu_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )
                prior_state_logvar_view = prior_state_logvar_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_state_feat_dim,
                )
                posterior_raw_nll_t, posterior_nll_t = self._stabilized_gaussian_nll(
                    hidden_enemy_view,
                    posterior_state_mu_view,
                    posterior_state_logvar_view,
                )
                prior_raw_nll_t, prior_nll_t = self._stabilized_gaussian_nll(
                    hidden_enemy_view,
                    prior_state_mu_view,
                    prior_state_logvar_view,
                )
                belief_recon_sum = belief_recon_sum + (posterior_nll_t * hidden_mask_t).sum()
                belief_recon_denom = belief_recon_denom + hidden_mask_t.sum()
                belief_prior_nll_sum = belief_prior_nll_sum + (prior_nll_t * hidden_mask_t).sum()
                belief_prior_nll_denom = belief_prior_nll_denom + hidden_mask_t.sum()
                post_raw_nll_sum = post_raw_nll_sum + (posterior_raw_nll_t * hidden_mask_t).sum()
                post_raw_nll_denom = post_raw_nll_denom + hidden_mask_t.sum()
                prior_raw_nll_sum = prior_raw_nll_sum + (prior_raw_nll_t * hidden_mask_t).sum()
                prior_raw_nll_denom = prior_raw_nll_denom + hidden_mask_t.sum()

                kl_t = self._kl_divergence(posterior_z_mu_t, posterior_z_logvar_t, prior_z_mu_t, prior_z_logvar_t)
                belief_kl_sum = belief_kl_sum + (kl_t * hidden_present_t).sum()
                belief_kl_denom = belief_kl_denom + hidden_present_t.sum()

            if mac_value_active and hidden_enemy_view is not None and t < batch.max_seq_length - 1:
                valid_step_mask_t = mask[:, t, 0]
                valid_agent_mask_t = valid_step_mask_t.unsqueeze(1).expand(-1, self.args.n_agents)
                q_context_t = target_q_context_cache[t]
                visible_enemy_summary_t = target_visible_enemy_summary_cache[t]
                student_hidden_summary_t = self.mac.get_student_hidden_summary()
                oracle_hidden_summary_t, delta_q_target_t, student_delta_q_t, q_visible_t, q_student_t, q_oracle_t = self._build_oracle_targets(
                    q_context_t,
                    visible_enemy_summary_t,
                    hidden_enemy_view,
                    enemy_visible_t,
                    student_hidden_summary_t=student_hidden_summary_t,
                )
                distill_agent_mask_t = hidden_present_t * valid_agent_mask_t
                chosen_actions_t = actions[:, t]
                delta_q_target_chosen_t = self._gather_action_values(delta_q_target_t, chosen_actions_t)
                student_delta_q_chosen_t = self._gather_action_values(student_delta_q_t, chosen_actions_t)
                distill_importance_t = self._build_distill_importance(delta_q_target_chosen_t, distill_agent_mask_t)
                oracle_delta_abs_t = delta_q_target_chosen_t.abs()
                belief_action_help_t, belief_action_drag_t, belief_action_net_gain_t, belief_delta_cosine_t = (
                    self._belief_action_diagnostics(q_visible_t, q_student_t, q_oracle_t)
                )

                repr_error_t = F.smooth_l1_loss(
                    student_hidden_summary_t,
                    oracle_hidden_summary_t,
                    reduction="none",
                ).mean(dim=-1)
                repr_loss_sum = repr_loss_sum + (repr_error_t * distill_importance_t).sum()
                repr_loss_denom = repr_loss_denom + distill_importance_t.sum()

                aux_error_t = F.smooth_l1_loss(
                    student_delta_q_chosen_t,
                    delta_q_target_chosen_t.detach(),
                    reduction="none",
                )
                aux_delta_q_loss_sum = aux_delta_q_loss_sum + (aux_error_t * distill_importance_t).sum()
                aux_delta_q_loss_denom = aux_delta_q_loss_denom + distill_importance_t.sum()
                distill_importance_sum = distill_importance_sum + distill_importance_t.sum()
                distill_importance_count = distill_importance_count + distill_agent_mask_t.sum()
                oracle_delta_abs_sum = oracle_delta_abs_sum + (oracle_delta_abs_t * distill_agent_mask_t).sum()
                oracle_delta_abs_denom = oracle_delta_abs_denom + distill_agent_mask_t.sum()
                belief_action_help_sum = belief_action_help_sum + (belief_action_help_t * distill_agent_mask_t).sum()
                belief_action_drag_sum = belief_action_drag_sum + (belief_action_drag_t * distill_agent_mask_t).sum()
                belief_action_net_gain_sum = belief_action_net_gain_sum + (belief_action_net_gain_t * distill_agent_mask_t).sum()
                belief_action_metric_denom = belief_action_metric_denom + distill_agent_mask_t.sum()
                belief_delta_cosine_sum = belief_delta_cosine_sum + (belief_delta_cosine_t * distill_agent_mask_t).sum()
                belief_delta_cosine_denom = belief_delta_cosine_denom + distill_agent_mask_t.sum()

        mac_out = th.stack(mac_out, dim=1)

        if q_learning_active:
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            if self.mixer is not None:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                with th.no_grad():
                    target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            n_step = getattr(self.args, "n_step", 1)
            if n_step == 1:
                targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
            else:
                n_rewards = th.zeros_like(rewards)
                gamma_tensor = th.tensor(
                    [self.args.gamma ** i for i in range(n_step)],
                    dtype=th.float,
                    device=rewards.device,
                )
                steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(n_step).long()
                for i in range(batch.max_seq_length - 1):
                    horizon = min(n_step, batch.max_seq_length - 1 - i)
                    n_rewards[:, i, 0] = ((rewards * mask)[:, i:i + horizon, 0] * gamma_tensor[:horizon]).sum(dim=1)
                indices = th.arange(batch.max_seq_length - 1, device=steps.device).unsqueeze(1)
                n_targets_terminated = th.gather(target_max_qvals * (1 - terminated), dim=1, index=steps + indices - 1)
                targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

            td_error = chosen_action_qvals - targets.detach()
            q_mask = mask.expand_as(td_error)
            masked_td_error = td_error * q_mask
            q_loss = (masked_td_error ** 2).sum() / q_mask.sum()
        else:
            chosen_action_qvals = None
            targets = None
            td_error = None
            q_mask = None
            masked_td_error = None
            q_loss = rewards.new_tensor(0.0)

        belief_post_loss = belief_recon_sum / belief_recon_denom.clamp(min=1.0)
        belief_prior_loss = belief_prior_nll_sum / belief_prior_nll_denom.clamp(min=1.0)
        belief_post_raw_nll = post_raw_nll_sum / post_raw_nll_denom.clamp(min=1.0)
        belief_prior_raw_nll = prior_raw_nll_sum / prior_raw_nll_denom.clamp(min=1.0)
        belief_kl_loss = belief_kl_sum / belief_kl_denom.clamp(min=1.0)
        repr_loss = repr_loss_sum / repr_loss_denom.clamp(min=1.0)
        aux_delta_q_loss = aux_delta_q_loss_sum / aux_delta_q_loss_denom.clamp(min=1.0)
        distill_importance_mean = distill_importance_sum / distill_importance_count.clamp(min=1.0)
        oracle_delta_abs_mean = oracle_delta_abs_sum / oracle_delta_abs_denom.clamp(min=1.0)
        belief_action_help_rate = belief_action_help_sum / belief_action_metric_denom.clamp(min=1.0)
        belief_action_drag_rate = belief_action_drag_sum / belief_action_metric_denom.clamp(min=1.0)
        belief_action_net_gain = belief_action_net_gain_sum / belief_action_metric_denom.clamp(min=1.0)
        belief_delta_cosine = belief_delta_cosine_sum / belief_delta_cosine_denom.clamp(min=1.0)

        repr_weight = self.belief_repr_coef * value_warm if mac_value_active else 0.0
        aux_delta_q_weight = self.belief_aux_delta_q_coef * value_warm if mac_value_active else 0.0
        kl_progress = min(1.0, float(max(1, t_env)) / float(max(1, self.belief_kl_warmup_t)))
        belief_kl_weight = self.belief_kl_coef * self._smooth_ramp(kl_progress)
        if self.belief_pretrain_only:
            if t_env < self.belief_pretrain_t:
                belief_weight = self._smooth_ramp(float(max(0, t_env)) / float(max(1, self.belief_pretrain_t)))
            else:
                post_progress = min(
                    1.0,
                    float(max(0, t_env - self.belief_pretrain_t)) / float(max(1, self.belief_warmup_t)),
                )
                belief_weight = self.belief_loss_coef + (1.0 - self.belief_loss_coef) * (1.0 - self._smooth_ramp(post_progress))
        else:
            warmup_progress = min(1.0, float(max(0, t_env)) / float(max(1, self.belief_warmup_t)))
            belief_weight = self.belief_loss_coef * self._smooth_ramp(warmup_progress) if self.belief_loss_coef > 0.0 else 0.0
        belief_loss = belief_prior_loss + self.belief_posterior_coef * belief_post_loss + belief_kl_weight * belief_kl_loss
        belief_weighted_loss = belief_weight * belief_loss
        belief_effective_post_weight = belief_weight * self.belief_posterior_coef
        belief_effective_kl_weight = belief_weight * belief_kl_weight
        main_loss = q_loss + belief_weighted_loss + repr_weight * repr_loss + aux_delta_q_weight * aux_delta_q_loss

        if (
            not q_learning_active
            and belief_weight == 0.0
            and repr_weight == 0.0
            and aux_delta_q_weight == 0.0
        ):
            return

        self.optimiser.zero_grad()

        grad_norm = rewards.new_tensor(0.0)
        main_update_active = q_learning_active or belief_weight > 0.0 or repr_weight > 0.0 or aux_delta_q_weight > 0.0

        if main_update_active:
            main_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.online_params, self.args.grad_norm_clip)
            self.optimiser.step()

        loss = main_loss

        if q_learning_active and (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("main_loss", main_loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_weighted_loss", belief_weighted_loss.item(), t_env)
            self.logger.log_stat(
                "belief_weighted_q_ratio",
                abs(belief_weighted_loss.item()) / (q_loss.item() + 1e-6) if q_learning_active else 0.0,
                t_env,
            )
            self.logger.log_stat("belief_recon_loss", belief_post_loss.item(), t_env)
            self.logger.log_stat("belief_post_loss", belief_post_loss.item(), t_env)
            self.logger.log_stat("belief_prior_nll", belief_prior_loss.item(), t_env)
            self.logger.log_stat("belief_prior_loss", belief_prior_loss.item(), t_env)
            self.logger.log_stat("belief_post_raw_nll", belief_post_raw_nll.item(), t_env)
            self.logger.log_stat("belief_prior_raw_nll", belief_prior_raw_nll.item(), t_env)
            self.logger.log_stat("belief_kl_loss", belief_kl_loss.item(), t_env)
            self.logger.log_stat("belief_kl_weight", belief_kl_weight, t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_effective_post_weight", belief_effective_post_weight, t_env)
            self.logger.log_stat("belief_effective_kl_weight", belief_effective_kl_weight, t_env)
            self.logger.log_stat("belief_repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("belief_repr_weight", repr_weight, t_env)
            self.logger.log_stat("belief_aux_delta_q_loss", aux_delta_q_loss.item(), t_env)
            self.logger.log_stat("belief_aux_delta_q_weight", aux_delta_q_weight, t_env)
            self.logger.log_stat("belief_distill_importance_mean", distill_importance_mean.item(), t_env)
            self.logger.log_stat("belief_oracle_delta_abs_mean", oracle_delta_abs_mean.item(), t_env)
            self.logger.log_stat("belief_action_help_rate", belief_action_help_rate.item(), t_env)
            self.logger.log_stat("belief_action_drag_rate", belief_action_drag_rate.item(), t_env)
            self.logger.log_stat("belief_action_net_gain", belief_action_net_gain.item(), t_env)
            self.logger.log_stat("belief_delta_cosine", belief_delta_cosine.item(), t_env)
            self.logger.log_stat("belief_pretraining_active", float(belief_pretraining_active), t_env)
            self.logger.log_stat("belief_pretrain_only", float(self.belief_pretrain_only), t_env)
            self.logger.log_stat("belief_prior_state_mu_mean", prior_state_mu_sum.item() / prior_state_mu_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_state_logvar_mean", prior_state_logvar_sum.item() / prior_state_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_z_mu_mean", prior_z_mu_sum.item() / prior_z_mu_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_z_logvar_mean", prior_z_logvar_sum.item() / prior_z_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("belief_prior_confidence_mean", prior_conf_sum.item() / prior_conf_count.clamp(min=1.0).item(), t_env)
            if self.posterior_enabled or belief_pretraining_active:
                self.logger.log_stat("belief_post_state_mu_mean", post_state_mu_sum.item() / post_state_mu_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_state_logvar_mean", post_state_logvar_sum.item() / post_state_logvar_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_z_mu_mean", post_z_mu_sum.item() / post_z_mu_count.clamp(min=1.0).item(), t_env)
                self.logger.log_stat("belief_post_z_logvar_mean", post_z_logvar_sum.item() / post_z_logvar_count.clamp(min=1.0).item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if q_learning_active:
                mask_elems = q_mask.sum().item()
                self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
                self.logger.log_stat(
                    "q_taken_mean",
                    (chosen_action_qvals * q_mask).sum().item() / (mask_elems * self.args.n_agents),
                    t_env,
                )
                self.logger.log_stat(
                    "target_mean",
                    (targets * q_mask).sum().item() / (mask_elems * self.args.n_agents),
                    t_env,
                )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        opt_state = None
        opt_path = "{}/opt.th".format(path)
        if os.path.exists(opt_path):
            opt_state = th.load(opt_path, map_location=lambda storage, loc: storage)
            if not self._load_optimizer_state(self.optimiser, opt_state):
                online_opt_state = self._optimizer_state_subset(opt_state, 0, len(self.online_params))
                self._load_optimizer_state(self.optimiser, online_opt_state)
