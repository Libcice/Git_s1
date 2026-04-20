import copy
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

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
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
        self.belief_kl_coef = getattr(self.args, "belief_kl_coef", 0.1)
        self.belief_logvar_min = getattr(self.args, "belief_logvar_min", -3.0)
        self.belief_logvar_max = getattr(self.args, "belief_logvar_max", 1.0)
        self.belief_warmup_t = getattr(self.args, "belief_warmup_t", 300000)
        self.belief_pretrain_t = getattr(self.args, "belief_pretrain_t", 50000)
        self.belief_pretrain_only = getattr(self.args, "belief_pretrain_only", False)

        self.belief_teacher_q_coef = getattr(self.args, "belief_teacher_q_coef", 0.1)
        self.belief_repr_coef = getattr(self.args, "belief_repr_coef", 0.05)
        self.belief_aux_delta_q_coef = getattr(self.args, "belief_aux_delta_q_coef", 0.1)
        self.belief_value_warmup_t = getattr(self.args, "belief_value_warmup_t", 300000)

        self.disable_belief = getattr(self.args, "disable_belief", False)
        self.use_belief_for_q = getattr(self.args, "use_belief_for_q", True) and not self.disable_belief
        self.belief_prior_type = getattr(self.args, "belief_prior_type", "conditional")
        if self.disable_belief:
            self.belief_loss_coef = 0.0
            self.belief_kl_coef = 0.0
            self.belief_teacher_q_coef = 0.0
            self.belief_repr_coef = 0.0
            self.belief_aux_delta_q_coef = 0.0
            self.belief_pretrain_only = False

        self.posterior_enabled = self.belief_loss_coef > 0.0
        self.value_losses_enabled = (
            self.belief_teacher_q_coef > 0.0
            or self.belief_repr_coef > 0.0
            or self.belief_aux_delta_q_coef > 0.0
        )
        self.belief_enabled = self.posterior_enabled or self.use_belief_for_q or self.value_losses_enabled
        self.q_learning_started = False

        teacher_enemy_input_dim = self.enemy_state_feat_dim + self.ally_state_feat_dim * 2 + 1
        self.teacher_enemy_encoder = th.nn.Sequential(
            th.nn.Linear(teacher_enemy_input_dim, self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        teacher_query_input_dim = self.ally_state_feat_dim * 2 + self.enemy_state_feat_dim + 2
        self.teacher_query_encoder = th.nn.Sequential(
            th.nn.Linear(teacher_query_input_dim, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.teacher_delta_q_head = th.nn.Sequential(
            th.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            th.nn.ReLU(),
            th.nn.Linear(self.hidden_dim * 2, self.args.n_actions),
        )
        self.params += list(self.teacher_enemy_encoder.parameters())
        self.params += list(self.teacher_query_encoder.parameters())
        self.params += list(self.teacher_delta_q_head.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _gaussian_nll(self, target, mu, logvar):
        logvar = logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        inv_var = th.exp(-logvar)
        return 0.5 * (((target - mu) ** 2) * inv_var + logvar).sum(dim=-1)

    def _kl_divergence(self, post_mu, post_logvar, prior_mu, prior_logvar):
        post_logvar = post_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        prior_logvar = prior_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)

        if self.belief_prior_type == "standard_normal":
            return 0.5 * (th.exp(post_logvar) + post_mu.pow(2) - 1.0 - post_logvar).sum(dim=-1)

        diff = post_mu - prior_mu
        return 0.5 * (
            prior_logvar - post_logvar + (th.exp(post_logvar) + diff.pow(2)) * th.exp(-prior_logvar) - 1.0
        ).sum(dim=-1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        belief_pretraining_active = (self.belief_loss_coef > 0.0) and (t_env < self.belief_pretrain_t)
        q_learning_active = (not self.belief_pretrain_only) or (t_env >= self.belief_pretrain_t)
        if q_learning_active and not self.q_learning_started:
            self._update_targets()
            self.q_learning_started = True

        mac_value_active = q_learning_active and self.value_losses_enabled
        hidden_enemy_state_active = self.posterior_enabled or belief_pretraining_active or mac_value_active

        mac_out = []
        teacher_chosen_q_list = []
        teacher_global_mask_list = []

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
        teacher_q_loss = rewards.new_tensor(0.0)
        teacher_q_weight = 0.0
        repr_weight = 0.0
        aux_delta_q_weight = 0.0
        value_warm = min(1.0, float(t_env) / float(max(1, self.belief_value_warmup_t)))
        belief_warm = 1.0 if belief_pretraining_active else min(
            1.0,
            float(max(1, t_env - self.belief_pretrain_t + 1)) / float(max(1, self.belief_warmup_t)),
        )

        enemy_obs_start = self.move_dim
        enemy_obs_end = enemy_obs_start + self.args.enemy_num * self.enemy_obs_feat_dim

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
            prior_belief_feat_t = self.mac.get_prior_belief_feat()
            aux_belief_values_t = self.mac.get_aux_belief_values()
            q_visible_t = self.mac.get_q_visible()

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

                enemy_obs_t = batch["obs"][:, t, :, enemy_obs_start:enemy_obs_end]
                enemy_obs_t = enemy_obs_t.view(
                    batch.batch_size,
                    self.args.n_agents,
                    self.args.enemy_num,
                    self.enemy_obs_feat_dim,
                )
                enemy_visible_t = (enemy_obs_t.abs().sum(dim=-1) > 0).float()
                alive_mask_t = (
                    hidden_enemy_state_t.view(
                        batch.batch_size,
                        self.args.n_agents,
                        self.args.enemy_num,
                        self.enemy_state_feat_dim,
                    )[..., 0]
                    > 0
                ).float()
                time_mask_t = batch["filled"][:, t].float().unsqueeze(1).expand(-1, self.args.n_agents, -1)
                hidden_mask_t = (1.0 - enemy_visible_t) * alive_mask_t * time_mask_t
                hidden_present_t = (hidden_mask_t.sum(dim=-1) > 0).float()

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
                hidden_enemy_view = hidden_enemy_state_t.view(
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
                posterior_nll_t = self._gaussian_nll(hidden_enemy_view, posterior_state_mu_view, posterior_state_logvar_view)
                prior_nll_t = self._gaussian_nll(hidden_enemy_view, prior_state_mu_view, prior_state_logvar_view)
                belief_recon_sum = belief_recon_sum + (posterior_nll_t * hidden_mask_t).sum()
                belief_recon_denom = belief_recon_denom + hidden_mask_t.sum()
                belief_prior_nll_sum = belief_prior_nll_sum + (prior_nll_t * hidden_mask_t).sum()
                belief_prior_nll_denom = belief_prior_nll_denom + hidden_mask_t.sum()

                kl_t = self._kl_divergence(posterior_z_mu_t, posterior_z_logvar_t, prior_z_mu_t, prior_z_logvar_t)
                belief_kl_sum = belief_kl_sum + (kl_t * hidden_present_t).sum()
                belief_kl_denom = belief_kl_denom + hidden_present_t.sum()

                if mac_value_active and t < batch.max_seq_length - 1:
                    valid_step_mask_t = mask[:, t].reshape(batch.batch_size, -1)[:, 0]
                    valid_agent_mask_t = valid_step_mask_t.unsqueeze(1).expand(-1, self.args.n_agents)

                    state_target_t = hidden_enemy_view
                    ally_state_t = batch["state"][:, t, :self.ally_state_dim]
                    ally_state_t = ally_state_t.reshape(batch.batch_size, self.args.n_agents, self.ally_state_feat_dim)
                    own_state_t = ally_state_t
                    ally_mean_agent_t = ally_state_t.mean(dim=1, keepdim=True).expand(-1, self.args.n_agents, -1)
                    visible_enemy_mean_t = self._masked_mean(
                        state_target_t.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num, self.enemy_state_feat_dim),
                        enemy_visible_t.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num),
                    ).reshape(batch.batch_size, self.args.n_agents, self.enemy_state_feat_dim)
                    visible_frac_t = enemy_visible_t.mean(dim=-1, keepdim=True)
                    hidden_frac_t = hidden_mask_t.mean(dim=-1, keepdim=True)

                    teacher_query_t = self.teacher_query_encoder(
                        th.cat(
                            [own_state_t, ally_mean_agent_t, visible_enemy_mean_t, visible_frac_t, hidden_frac_t],
                            dim=-1,
                        )
                    )
                    teacher_query_expand_t = teacher_query_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                    own_state_expand_t = own_state_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                    ally_mean_expand_t = ally_mean_agent_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                    teacher_enemy_input_t = th.cat(
                        [state_target_t, own_state_expand_t, ally_mean_expand_t, enemy_visible_t.unsqueeze(-1)],
                        dim=-1,
                    )
                    teacher_enemy_repr_t = self.teacher_enemy_encoder(teacher_enemy_input_t)
                    teacher_delta_q_t = self.teacher_delta_q_head(
                        th.cat([teacher_query_expand_t, teacher_enemy_repr_t], dim=-1)
                    )
                    teacher_delta_q_hidden_t = (hidden_mask_t.unsqueeze(-1) * teacher_delta_q_t).sum(dim=2)

                    repr_error_t = F.smooth_l1_loss(
                        prior_belief_feat_t,
                        teacher_enemy_repr_t.detach(),
                        reduction="none",
                    ).mean(dim=-1)
                    distill_mask_t = hidden_mask_t * valid_agent_mask_t.unsqueeze(-1)
                    repr_loss_sum = repr_loss_sum + (repr_error_t * distill_mask_t).sum()
                    repr_loss_denom = repr_loss_denom + distill_mask_t.sum()

                    aux_error_t = F.smooth_l1_loss(
                        aux_belief_values_t,
                        teacher_delta_q_t.detach(),
                        reduction="none",
                    ).mean(dim=-1)
                    aux_delta_q_loss_sum = aux_delta_q_loss_sum + (aux_error_t * distill_mask_t).sum()
                    aux_delta_q_loss_denom = aux_delta_q_loss_denom + distill_mask_t.sum()

                    teacher_full_q_t = q_visible_t.detach() + teacher_delta_q_hidden_t
                    teacher_chosen_q_t = th.gather(teacher_full_q_t, dim=2, index=actions[:, t]).squeeze(2)
                    teacher_chosen_q_list.append(teacher_chosen_q_t)
                    teacher_global_mask_list.append((hidden_mask_t.sum(dim=-1) > 0).float())

            elif mac_value_active and t < batch.max_seq_length - 1:
                valid_step_mask_t = mask[:, t].reshape(batch.batch_size, -1)[:, 0]
                valid_agent_mask_t = valid_step_mask_t.unsqueeze(1).expand(-1, self.args.n_agents)

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
                hidden_mask_t = (1.0 - enemy_visible_t) * alive_mask_t

                ally_state_t = batch["state"][:, t, :self.ally_state_dim]
                ally_state_t = ally_state_t.reshape(batch.batch_size, self.args.n_agents, self.ally_state_feat_dim)
                own_state_t = ally_state_t
                ally_mean_agent_t = ally_state_t.mean(dim=1, keepdim=True).expand(-1, self.args.n_agents, -1)
                visible_enemy_mean_t = self._masked_mean(
                    hidden_enemy_view.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num, self.enemy_state_feat_dim),
                    enemy_visible_t.reshape(batch.batch_size * self.args.n_agents, self.args.enemy_num),
                ).reshape(batch.batch_size, self.args.n_agents, self.enemy_state_feat_dim)
                visible_frac_t = enemy_visible_t.mean(dim=-1, keepdim=True)
                hidden_frac_t = hidden_mask_t.mean(dim=-1, keepdim=True)

                teacher_query_t = self.teacher_query_encoder(
                    th.cat(
                        [own_state_t, ally_mean_agent_t, visible_enemy_mean_t, visible_frac_t, hidden_frac_t],
                        dim=-1,
                    )
                )
                teacher_query_expand_t = teacher_query_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                own_state_expand_t = own_state_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                ally_mean_expand_t = ally_mean_agent_t.unsqueeze(2).expand(-1, -1, self.args.enemy_num, -1)
                teacher_enemy_input_t = th.cat(
                    [hidden_enemy_view, own_state_expand_t, ally_mean_expand_t, enemy_visible_t.unsqueeze(-1)],
                    dim=-1,
                )
                teacher_enemy_repr_t = self.teacher_enemy_encoder(teacher_enemy_input_t)
                teacher_delta_q_t = self.teacher_delta_q_head(
                    th.cat([teacher_query_expand_t, teacher_enemy_repr_t], dim=-1)
                )
                teacher_delta_q_hidden_t = (hidden_mask_t.unsqueeze(-1) * teacher_delta_q_t).sum(dim=2)

                repr_error_t = F.smooth_l1_loss(
                    prior_belief_feat_t,
                    teacher_enemy_repr_t.detach(),
                    reduction="none",
                ).mean(dim=-1)
                distill_mask_t = hidden_mask_t * valid_agent_mask_t.unsqueeze(-1)
                repr_loss_sum = repr_loss_sum + (repr_error_t * distill_mask_t).sum()
                repr_loss_denom = repr_loss_denom + distill_mask_t.sum()

                aux_error_t = F.smooth_l1_loss(
                    aux_belief_values_t,
                    teacher_delta_q_t.detach(),
                    reduction="none",
                ).mean(dim=-1)
                aux_delta_q_loss_sum = aux_delta_q_loss_sum + (aux_error_t * distill_mask_t).sum()
                aux_delta_q_loss_denom = aux_delta_q_loss_denom + distill_mask_t.sum()

                teacher_full_q_t = q_visible_t.detach() + teacher_delta_q_hidden_t
                teacher_chosen_q_t = th.gather(teacher_full_q_t, dim=2, index=actions[:, t]).squeeze(2)
                teacher_chosen_q_list.append(teacher_chosen_q_t)
                teacher_global_mask_list.append((hidden_mask_t.sum(dim=-1) > 0).float())

        mac_out = th.stack(mac_out, dim=1)

        if q_learning_active:
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t)
                    target_mac_out.append(target_agent_outs)
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

        if teacher_chosen_q_list and q_learning_active:
            teacher_chosen_action_qvals = th.stack(teacher_chosen_q_list, dim=1)
            teacher_global_mask = th.stack(teacher_global_mask_list, dim=1)
            if self.mixer is not None:
                teacher_chosen_action_qvals = self.mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])
                teacher_global_mask = (teacher_global_mask.sum(dim=-1, keepdim=True) > 0).float()
            teacher_td_error = teacher_chosen_action_qvals - targets.detach()
            teacher_masked_td_error = teacher_td_error * q_mask
            teacher_q_loss = (
                ((teacher_masked_td_error ** 2) * teacher_global_mask).sum()
                / teacher_global_mask.sum().clamp(min=1.0)
            )
            teacher_q_weight = self.belief_teacher_q_coef * value_warm

        belief_recon_loss = belief_recon_sum / belief_recon_denom.clamp(min=1.0)
        belief_prior_nll = belief_prior_nll_sum / belief_prior_nll_denom.clamp(min=1.0)
        belief_kl_loss = belief_kl_sum / belief_kl_denom.clamp(min=1.0)
        belief_loss = belief_recon_loss + self.belief_kl_coef * belief_kl_loss
        repr_loss = repr_loss_sum / repr_loss_denom.clamp(min=1.0)
        aux_delta_q_loss = aux_delta_q_loss_sum / aux_delta_q_loss_denom.clamp(min=1.0)

        if not q_learning_active:
            teacher_q_weight = 0.0
        repr_weight = self.belief_repr_coef * value_warm if mac_value_active else 0.0
        aux_delta_q_weight = self.belief_aux_delta_q_coef * value_warm if mac_value_active else 0.0
        belief_weight = self.belief_loss_coef * belief_warm if self.belief_loss_coef > 0.0 else 0.0

        if not q_learning_active and belief_weight == 0.0 and teacher_q_weight == 0.0 and repr_weight == 0.0 and aux_delta_q_weight == 0.0:
            return

        loss = (
            q_loss
            + belief_weight * belief_loss
            + teacher_q_weight * teacher_q_loss
            + repr_weight * repr_loss
            + aux_delta_q_weight * aux_delta_q_loss
        )

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if q_learning_active and (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("belief_loss", belief_loss.item(), t_env)
            self.logger.log_stat("belief_recon_loss", belief_recon_loss.item(), t_env)
            self.logger.log_stat("belief_prior_nll", belief_prior_nll.item(), t_env)
            self.logger.log_stat("belief_kl_loss", belief_kl_loss.item(), t_env)
            self.logger.log_stat("belief_weight", belief_weight, t_env)
            self.logger.log_stat("belief_teacher_q_loss", teacher_q_loss.item(), t_env)
            self.logger.log_stat("belief_teacher_q_weight", teacher_q_weight, t_env)
            self.logger.log_stat("belief_repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("belief_repr_weight", repr_weight, t_env)
            self.logger.log_stat("belief_aux_delta_q_loss", aux_delta_q_loss.item(), t_env)
            self.logger.log_stat("belief_aux_delta_q_weight", aux_delta_q_weight, t_env)
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
        self.teacher_enemy_encoder.cuda()
        self.teacher_query_encoder.cuda()
        self.teacher_delta_q_head.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.teacher_enemy_encoder.state_dict(), "{}/teacher_enemy_encoder.th".format(path))
        th.save(self.teacher_query_encoder.state_dict(), "{}/teacher_query_encoder.th".format(path))
        th.save(self.teacher_delta_q_head.state_dict(), "{}/teacher_delta_q_head.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        enemy_path = "{}/teacher_enemy_encoder.th".format(path)
        if os.path.exists(enemy_path):
            self.teacher_enemy_encoder.load_state_dict(
                th.load(enemy_path, map_location=lambda storage, loc: storage)
            )
        query_path = "{}/teacher_query_encoder.th".format(path)
        if os.path.exists(query_path):
            self.teacher_query_encoder.load_state_dict(
                th.load(query_path, map_location=lambda storage, loc: storage)
            )
        delta_path = "{}/teacher_delta_q_head.th".format(path)
        if os.path.exists(delta_path):
            self.teacher_delta_q_head.load_state_dict(
                th.load(delta_path, map_location=lambda storage, loc: storage)
            )
        opt_state = th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        try:
            self.optimiser.load_state_dict(opt_state)
        except ValueError:
            pass
