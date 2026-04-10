import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gym.spaces import Box
except ImportError:
    from gymnasium.spaces import Box

from mappo_onpolicy.models.act import ACTLayer
from mappo_onpolicy.models.actor_critic import R_Critic
from mappo_onpolicy.models.distributions import Categorical, FixedCategorical
from mappo_onpolicy.models.mlp import MLPBase
from mappo_onpolicy.models.rnn import RNNLayer
from mappo_onpolicy.models.util import check
from mappo_onpolicy.utils.util import get_shape_from_obs_space
from mappo_belief.models.smac_tokenizer import SMACTokenizer
from mappo_belief.models.trxl import build_trxl_encoder


def _augment_box_space(space, context_dim):
    shape = tuple(space.shape)
    if len(shape) != 1:
        raise NotImplementedError("BeliefAwareCritic currently expects flat centralized observations")
    aug_shape = (shape[0] + context_dim,)
    low = np.full(aug_shape, -np.inf, dtype=np.float32)
    high = np.full(aug_shape, np.inf, dtype=np.float32)
    return Box(low=low, high=high, dtype=np.float32)


class BeliefAwareCritic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.use_belief_critic = getattr(args, "use_belief_critic", False)
        self.context_dim = args.hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        critic_space = _augment_box_space(cent_obs_space, self.context_dim) if self.use_belief_critic else cent_obs_space
        self.critic = R_Critic(args, critic_space, device=device)
        self.v_out = self.critic.v_out
        self.context_adapter = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, self.context_dim),
            nn.ReLU(),
        )
        self.to(device)

    def forward(self, cent_obs, critic_context, rnn_states, masks):
        if not self.use_belief_critic:
            return self.critic(cent_obs, rnn_states, masks)

        cent_obs = check(cent_obs).to(**self.tpdv)
        if critic_context is None:
            raise ValueError("critic_context is required when use_belief_critic=True")
        critic_context = check(critic_context).to(**self.tpdv).detach()
        critic_context = self.context_adapter(critic_context)
        critic_input = torch.cat([cent_obs, critic_context], dim=-1)
        return self.critic(critic_input, rnn_states, masks)


class BeliefTransformerActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.tokenizer = SMACTokenizer(args, device)
        self.layout = self.tokenizer.layout
        self.token_dim = self.layout.token_dim
        # Store the recent K-1 tokenized observations inside the actor state so
        # rollout-time action selection and sequence training see the same window.
        self.history_steps = max(int(getattr(args, "history_steps", 1)), 1)
        self.history_state_steps = max(self.history_steps - 1, 0)
        self.history_state_dim = self.history_state_steps * self.layout.tokens_per_step * self.token_dim
        # Optional latent belief mode keeps an explicit stochastic state z_t in
        # the recurrent actor state instead of relying on deterministic memory alone.
        self.use_latent_belief = getattr(args, "use_latent_belief", False)
        self.latent_belief_dim = int(getattr(args, "latent_belief_dim", self.hidden_size))
        self.latent_state_dim = self.latent_belief_dim if self.use_latent_belief else 0
        self.base_actor_state_size = self.hidden_size + self.latent_state_dim
        self.actor_state_size = self.base_actor_state_size + self.history_state_dim
        # "self_attention_token" reproduces the old memory-token encoder path.
        # "cross_attention" first encodes current-step tokens, then lets memory
        # read them explicitly as a query.
        self.memory_update_mode = getattr(args, "belief_memory_update", "self_attention_token")
        if self.memory_update_mode not in {"self_attention_token", "cross_attention"}:
            raise ValueError(
                "belief_memory_update must be 'self_attention_token' or 'cross_attention', got {}".format(
                    self.memory_update_mode
                )
            )
        self.n_enemies = args.enemy_num
        self.n_actions = getattr(args, "n_actions", action_space.n if action_space.__class__.__name__ == "Discrete" else None)
        self.n_actions_no_attack = (
            self.n_actions - self.n_enemies
            if self.n_actions is not None and self.n_enemies is not None
            else 6
        )
        self.enemy_state_feat_dim = self.layout.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 3)
        self.use_belief_gate = getattr(args, "use_belief_gate", True)
        self.belief_gate_warmup_t = getattr(args, "belief_gate_warmup_t", getattr(args, "belief_warmup_t", 500000))
        self.belief_confidence_temp = getattr(args, "belief_confidence_temp", 1.0)
        self.belief_logvar_min = getattr(args, "belief_logvar_min", -2.0)
        self.belief_logvar_max = getattr(args, "belief_logvar_max", 1.0)
        self.latent_logvar_min = getattr(args, "latent_logvar_min", -4.0)
        self.latent_logvar_max = getattr(args, "latent_logvar_max", 2.0)
        self.detach_belief_to_policy = getattr(args, "detach_belief_to_policy", False)
        self.scheduled_detach_belief = getattr(args, "scheduled_detach_belief", False)
        self.detach_belief_until_t = getattr(args, "detach_belief_until_t", 1500000)
        self.detach_belief_release_t = getattr(args, "detach_belief_release_t", 2500000)
        self.belief_policy_alpha = getattr(args, "belief_policy_alpha", 1.0)
        self.scheduled_belief_policy = getattr(args, "scheduled_belief_policy", False)
        self.belief_policy_until_t = getattr(args, "belief_policy_until_t", 0)
        self.belief_policy_release_t = getattr(args, "belief_policy_release_t", 1500000)

        self.token_embed = nn.Linear(self.token_dim, self.hidden_size)
        self.max_seq_len = 1 + self.history_steps * self.layout.tokens_per_step
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.hidden_size))
        self.transformer_heads = getattr(args, "transformer_heads", 4)
        self.transformer_dropout = getattr(args, "transformer_dropout", 0.0)
        self.trxl_ff_mult = getattr(args, "trxl_ff_mult", 4)
        self.encoder = build_trxl_encoder(
            hidden_dim=self.hidden_size,
            n_heads=self.transformer_heads,
            n_layers=getattr(args, "transformer_layers", 2),
            dropout=self.transformer_dropout,
            activation=getattr(args, "trxl_activation", "relu"),
            norm_first=getattr(args, "trxl_norm_first", False),
            final_norm=getattr(args, "trxl_final_norm", False),
            ff_mult=self.trxl_ff_mult,
            impl=getattr(args, "trxl_impl", "builtin"),
        )
        self.memory_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.transformer_heads,
            dropout=self.transformer_dropout,
            batch_first=True,
        )
        self.memory_cross_norm = nn.LayerNorm(self.hidden_size)
        self.memory_cross_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * self.trxl_ff_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_size * self.trxl_ff_mult, self.hidden_size),
        )
        self.memory_cross_ffn_norm = nn.LayerNorm(self.hidden_size)

        if self.use_latent_belief:
            self.prev_action_embed = nn.Linear(self.n_actions, self.hidden_size)
            self.latent_state_embed = nn.Linear(self.latent_belief_dim, self.hidden_size)
            self.latent_obs_proj = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.ReLU(),
            )
            self.latent_prior = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.latent_belief_dim * 2),
            )
            self.latent_post = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.latent_belief_dim * 2),
            )
            self.latent_to_memory = nn.Sequential(
                nn.Linear(self.latent_belief_dim, self.hidden_size),
                nn.ReLU(),
            )

        belief_out_dim = self.n_enemies * self.enemy_state_feat_dim
        self.belief_mu_head = nn.Linear(self.hidden_size, belief_out_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_size, belief_out_dim)
        self.belief_u_head = nn.Linear(
            self.hidden_size,
            belief_out_dim * self.belief_lowrank_rank,
        )

        self.belief_enemy_proj = nn.Linear(self.enemy_state_feat_dim, self.hidden_size)
        self.use_split_belief_logits = (
            action_space.__class__.__name__ == "Discrete"
            and not getattr(args, "use_residual_belief_actor", False)
        )
        # Old: flatten every enemy slot before the policy head.
        # self.policy_in_dim = self.hidden_size * (4 + self.n_enemies)
        # self.policy_proj = nn.Sequential(
        #     nn.Linear(self.policy_in_dim, self.hidden_size),
        #     nn.ReLU(),
        # )
        if self.use_split_belief_logits:
            self.base_policy_in_dim = self.hidden_size * 5
            self.base_policy_proj = nn.Sequential(
                nn.Linear(self.base_policy_in_dim, self.hidden_size),
                nn.ReLU(),
            )
            # Let belief bias only the attack-action slice instead of overriding the whole policy.
            self.belief_policy_in_dim = self.hidden_size * 2
            self.belief_policy_proj = nn.Sequential(
                nn.Linear(self.belief_policy_in_dim, self.hidden_size),
                nn.ReLU(),
            )
            self.base_action_out = Categorical(self.hidden_size, action_space.n, self._use_orthogonal, self._gain)
            self.belief_attack_out = Categorical(self.hidden_size, self.n_enemies, self._use_orthogonal, self._gain)
        else:
            # Old: read visible and unseen enemies as two summary streams.
            # self.policy_in_dim = self.hidden_size * 6
            # Keep visible and unseen enemy summaries separate for the MAPPO actor.
            self.policy_in_dim = self.hidden_size * 6
            self.policy_proj = nn.Sequential(
                nn.Linear(self.policy_in_dim, self.hidden_size),
                nn.ReLU(),
            )
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        # Track gate progress for logging; when the gate is disabled this buffer stays at 1.0 for a clean control run.
        self.register_buffer("_belief_gate_alpha", torch.zeros(1, device=device))
        detach_init = 1.0 if (self.detach_belief_to_policy or self.scheduled_detach_belief) else 0.0
        self.register_buffer("_belief_detach_alpha", torch.full((1,), detach_init, device=device))
        policy_alpha_init = 0.0 if self.scheduled_belief_policy else float(self.belief_policy_alpha)
        self.register_buffer("_belief_policy_alpha", torch.full((1,), policy_alpha_init, device=device))
        self.to(device)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.actor_state_size)

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def set_belief_gate_progress(self, total_num_steps):
        if not self.use_belief_gate:
            alpha = 1.0
        elif self.belief_gate_warmup_t <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, float(total_num_steps) / float(self.belief_gate_warmup_t))
        self._belief_gate_alpha.fill_(alpha)

        if self.scheduled_detach_belief:
            detach_start = float(self.detach_belief_until_t)
            detach_end = float(max(self.detach_belief_release_t, self.detach_belief_until_t))
            if float(total_num_steps) <= detach_start:
                detach_alpha = 1.0
            elif float(total_num_steps) >= detach_end:
                detach_alpha = 0.0
            else:
                detach_alpha = 1.0 - ((float(total_num_steps) - detach_start) / max(detach_end - detach_start, 1.0))
        else:
            detach_alpha = 1.0 if self.detach_belief_to_policy else 0.0
        self._belief_detach_alpha.fill_(detach_alpha)

        if self.scheduled_belief_policy:
            policy_start = float(self.belief_policy_until_t)
            policy_end = float(max(self.belief_policy_release_t, self.belief_policy_until_t))
            if float(total_num_steps) <= policy_start:
                policy_alpha = 0.0
            elif float(total_num_steps) >= policy_end:
                policy_alpha = float(self.belief_policy_alpha)
            else:
                policy_alpha = float(self.belief_policy_alpha) * (
                    (float(total_num_steps) - policy_start) / max(policy_end - policy_start, 1.0)
                )
        else:
            policy_alpha = float(self.belief_policy_alpha)
        self._belief_policy_alpha.fill_(policy_alpha)

    def get_belief_gate_alpha(self):
        return float(self._belief_gate_alpha.item())

    def get_belief_detach_alpha(self):
        return float(self._belief_detach_alpha.item())

    def get_belief_policy_alpha(self):
        return float(self._belief_policy_alpha.item())

    def _policy_belief_alpha_tensor(self, ref_tensor):
        return self._belief_policy_alpha.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _empty_latent_state(self, batch_size, dtype, device):
        if not self.use_latent_belief:
            return None
        return torch.zeros(batch_size, self.latent_belief_dim, dtype=dtype, device=device)

    def _empty_history_tokens(self, batch_size, dtype, device):
        if self.history_state_steps <= 0:
            return None
        return torch.zeros(
            batch_size,
            self.history_state_steps,
            self.layout.tokens_per_step,
            self.token_dim,
            dtype=dtype,
            device=device,
        )

    def _clamp_latent_logvar(self, logvar):
        return logvar.clamp(min=self.latent_logvar_min, max=self.latent_logvar_max)

    def _split_latent_params(self, params):
        mu, logvar = params.chunk(2, dim=-1)
        return mu, self._clamp_latent_logvar(logvar)

    def _pack_actor_state(self, memory, latent_state, history_tokens):
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        if memory.size(1) == 1 and self._recurrent_N > 1:
            memory = memory.expand(-1, self._recurrent_N, -1)

        state_parts = [memory]

        if self.use_latent_belief:
            if latent_state is None:
                latent_flat = memory.new_zeros(memory.size(0), memory.size(1), self.latent_state_dim)
            else:
                if latent_state.dim() == 2:
                    latent_state = latent_state.unsqueeze(1)
                if latent_state.size(1) == 1 and self._recurrent_N > 1:
                    latent_state = latent_state.expand(-1, self._recurrent_N, -1)
                latent_flat = latent_state
            state_parts.append(latent_flat)

        if self.history_state_dim <= 0:
            return torch.cat(state_parts, dim=-1).contiguous()

        if history_tokens is None:
            history_flat = memory.new_zeros(memory.size(0), memory.size(1), self.history_state_dim)
        else:
            history_flat = history_tokens.reshape(history_tokens.size(0), 1, self.history_state_dim)
            if history_flat.size(1) == 1 and self._recurrent_N > 1:
                history_flat = history_flat.expand(-1, self._recurrent_N, -1)
        state_parts.append(history_flat)

        return torch.cat(state_parts, dim=-1).contiguous()

    def _unpack_actor_state(self, rnn_states):
        rnn_states = check(rnn_states).to(**self.tpdv)
        batch_size = rnn_states.size(0)
        old_actor_state_size = self.hidden_size + self.history_state_dim

        if rnn_states.size(-1) == self.hidden_size:
            memory = rnn_states[:, 0, :].contiguous()
            latent_state = self._empty_latent_state(batch_size, memory.dtype, memory.device)
            history_tokens = self._empty_history_tokens(batch_size, memory.dtype, memory.device)
            return memory, latent_state, history_tokens

        if self.use_latent_belief and rnn_states.size(-1) == old_actor_state_size:
            memory = rnn_states[:, 0, :self.hidden_size].contiguous()
            latent_state = self._empty_latent_state(batch_size, memory.dtype, memory.device)
            if self.history_state_dim <= 0:
                return memory, latent_state, None
            history_flat = rnn_states[:, 0, self.hidden_size:].contiguous()
            history_tokens = history_flat.view(
                batch_size,
                self.history_state_steps,
                self.layout.tokens_per_step,
                self.token_dim,
            )
            return memory, latent_state, history_tokens

        if rnn_states.size(-1) != self.actor_state_size:
            raise ValueError(
                "BeliefTransformerActor expected packed actor state size {}, got {}".format(
                    self.actor_state_size,
                    rnn_states.size(-1),
                )
            )

        memory = rnn_states[:, 0, :self.hidden_size].contiguous()
        offset = self.hidden_size
        if self.use_latent_belief:
            latent_state = rnn_states[:, 0, offset:offset + self.latent_state_dim].contiguous()
            offset += self.latent_state_dim
        else:
            latent_state = None
        if self.history_state_dim <= 0:
            return memory, latent_state, None

        history_flat = rnn_states[:, 0, offset:].contiguous()
        history_tokens = history_flat.view(
            batch_size,
            self.history_state_steps,
            self.layout.tokens_per_step,
            self.token_dim,
        )
        return memory, latent_state, history_tokens

    def _compose_history_tokens(self, step_tokens, history_tokens):
        if self.history_state_steps <= 0:
            return step_tokens, None

        if history_tokens is None:
            history_tokens = self._empty_history_tokens(step_tokens.size(0), step_tokens.dtype, step_tokens.device)

        # Rebuild the explicit history window expected by the transformer, then
        # keep only the most recent K-1 steps in the recurrent actor state.
        full_history = torch.cat([history_tokens, step_tokens.unsqueeze(1)], dim=1)
        next_history = full_history[:, -self.history_state_steps:].contiguous()
        history_seq = full_history.reshape(step_tokens.size(0), -1, self.token_dim)
        return history_seq, next_history

    def _build_latent_belief(self, memory, new_memory, prev_actions, obs_summary, prev_latent_state):
        if not self.use_latent_belief:
            return new_memory, None

        if prev_latent_state is None:
            prev_latent_state = self._empty_latent_state(new_memory.size(0), new_memory.dtype, new_memory.device)
        if prev_actions is None:
            prev_action_feat = new_memory.new_zeros(new_memory.size(0), self.hidden_size)
        else:
            prev_action_feat = F.relu(self.prev_action_embed(prev_actions))

        prev_latent_feat = F.relu(self.latent_state_embed(prev_latent_state))
        prior_mu, prior_logvar = self._split_latent_params(
            self.latent_prior(torch.cat([memory, prev_latent_feat, prev_action_feat], dim=-1))
        )
        post_mu, post_logvar = self._split_latent_params(
            self.latent_post(torch.cat([new_memory, obs_summary, prev_latent_feat, prev_action_feat], dim=-1))
        )

        # Use the posterior mean for policy-time features so PPO re-evaluations
        # stay deterministic, while the latent distribution is still trained by KL.
        latent_state = post_mu
        latent_context = F.relu(self.latent_to_memory(latent_state))
        belief_context = new_memory + latent_context
        return belief_context, {
            "latent_state": latent_state,
            "latent_context": latent_context,
            "latent_prior_mu": prior_mu,
            "latent_prior_logvar": prior_logvar,
            "latent_post_mu": post_mu,
            "latent_post_logvar": post_logvar,
        }

    def _build_unseen_belief_mask(self, enemy_visible, belief_logvar):
        # Set use_belief_gate=False to recover the ungated baseline while keeping the rest of the actor unchanged.
        if not self.use_belief_gate:
            return 1.0 - enemy_visible
        belief_logvar = belief_logvar.clamp(min=self.belief_logvar_min, max=self.belief_logvar_max)
        confidence = torch.sigmoid(-belief_logvar.mean(dim=-1) / max(self.belief_confidence_temp, 1e-6))
        gate_alpha = self._belief_gate_alpha.to(device=enemy_visible.device, dtype=enemy_visible.dtype)
        return (1.0 - enemy_visible) * confidence * gate_alpha

    def _build_actor_enemy_context(
        self,
        visible_enemy_feat,
        belief_enemy_feat,
        enemy_visible,
        belief_logvar,
        policy_alpha,
    ):
        unseen_belief_mask = self._build_unseen_belief_mask(enemy_visible, belief_logvar)
        visible_enemy_summary = self._masked_mean(visible_enemy_feat, enemy_visible)
        unseen_enemy_summary = self._masked_mean(belief_enemy_feat, unseen_belief_mask)

        # Compatibility summary for logs and downstream code that still expects one enemy context tensor.
        policy_belief_enemy_feat = belief_enemy_feat * policy_alpha
        fused_enemy_feat = (
            enemy_visible.unsqueeze(-1) * visible_enemy_feat
            + unseen_belief_mask.unsqueeze(-1) * policy_belief_enemy_feat
        )
        enemy_summary_mask = enemy_visible + unseen_belief_mask
        enemy_summary = self._masked_mean(fused_enemy_feat, enemy_summary_mask)

        # Keep critic context on the full belief path so belief-critic ablations stay orthogonal to actor-only gating.
        critic_fused_enemy_feat = (
            enemy_visible.unsqueeze(-1) * visible_enemy_feat
            + unseen_belief_mask.unsqueeze(-1) * belief_enemy_feat
        )
        critic_context = self._masked_mean(critic_fused_enemy_feat, enemy_summary_mask)

        return {
            "actor_enemy_features": (visible_enemy_summary, unseen_enemy_summary),
            "visible_enemy_summary": visible_enemy_summary,
            "unseen_enemy_summary": unseen_enemy_summary,
            "enemy_summary": enemy_summary,
            "critic_context": critic_context,
        }

    def _update_memory(self, step_emb, memory):
        if self.memory_update_mode == "cross_attention":
            # Cross-attention variant: encode current-step tokens first, then let
            # memory query the encoded token set instead of acting as a normal token.
            token_seq = step_emb + self.pos_embed[:, 1:1 + step_emb.size(1)]
            encoded_tokens = self.encoder(token_seq)
            memory_query = memory.unsqueeze(1)
            attn_out, _ = self.memory_cross_attn(
                query=memory_query,
                key=encoded_tokens,
                value=encoded_tokens,
            )
            new_memory = self.memory_cross_norm(memory_query + attn_out)
            ffn_out = self.memory_cross_ffn(new_memory)
            new_memory = self.memory_cross_ffn_norm(new_memory + ffn_out)
            return new_memory.squeeze(1)

        # Old path: prepend memory as token 0 and update it through self-attention.
        seq = torch.cat([memory.unsqueeze(1), step_emb], dim=1)
        seq = seq + self.pos_embed[:, :seq.size(1)]
        encoded = self.encoder(seq)
        return encoded[:, 0, :]

    def _encode_step(self, obs, prev_actions, agent_ids, memory, latent_state, history_tokens):
        step_tokens, current = self.tokenizer(obs, prev_actions=prev_actions, agent_ids=agent_ids)
        history_seq, next_history = self._compose_history_tokens(step_tokens, history_tokens)
        history_emb = F.relu(self.token_embed(history_seq))
        new_memory = self._update_memory(history_emb, memory)

        move_feat = F.relu(self.token_embed(current["move_token"]))
        self_feat = F.relu(self.token_embed(current["self_token"]))
        ally_feat = self._masked_mean(
            F.relu(self.token_embed(current["ally_tokens"])),
            current["ally_visible"],
        )
        visible_enemy_feat = F.relu(self.token_embed(current["enemy_tokens"]))
        enemy_visible = current["enemy_visible"].float()
        visible_enemy_summary = self._masked_mean(visible_enemy_feat, enemy_visible)

        if self.use_latent_belief:
            obs_summary = self.latent_obs_proj(
                torch.cat([move_feat, self_feat, ally_feat, visible_enemy_summary], dim=-1)
            )
            belief_context, latent_aux = self._build_latent_belief(
                memory,
                new_memory,
                prev_actions,
                obs_summary,
                latent_state,
            )
            next_latent_state = latent_aux["latent_state"]
        else:
            belief_context = new_memory
            latent_aux = {}
            next_latent_state = None

        belief_mu = self.belief_mu_head(belief_context).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        belief_logvar = self.belief_logvar_head(belief_context).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        belief_u = self.belief_u_head(belief_context).view(
            -1, self.n_enemies, self.enemy_state_feat_dim, self.belief_lowrank_rank
        )
        # Scheduled detach keeps belief self-supervised early, then gradually hands policy gradients back later in training.
        detach_alpha = self._belief_detach_alpha.to(device=belief_mu.device, dtype=belief_mu.dtype)
        belief_actor_src = belief_mu.detach() * detach_alpha + belief_mu * (1.0 - detach_alpha)
        belief_enemy_feat = F.relu(self.belief_enemy_proj(belief_actor_src))
        policy_alpha = self._policy_belief_alpha_tensor(belief_mu)
        # Old: merge all enemies into one long vector.
        # fused_enemy_feat = enemy_visible.unsqueeze(-1) * visible_enemy_feat + (1.0 - enemy_visible.unsqueeze(-1)) * belief_enemy_feat
        # enemy_feat = fused_enemy_feat.flatten(start_dim=1)
        # Old: split visible and unseen enemies into two summary streams.
        # visible_enemy_summary = self._masked_mean(visible_enemy_feat, enemy_visible)
        # unseen_belief_mask = self._build_unseen_belief_mask(enemy_visible, belief_logvar)
        # unseen_enemy_summary = self._masked_mean(belief_enemy_feat, unseen_belief_mask)
        actor_enemy_context = self._build_actor_enemy_context(
            visible_enemy_feat,
            belief_enemy_feat,
            enemy_visible,
            belief_logvar,
            policy_alpha,
        )

        if self.use_split_belief_logits:
            base_actor_features = self.base_policy_proj(
                torch.cat(
                    [
                        belief_context,
                        move_feat,
                        self_feat,
                        ally_feat,
                        actor_enemy_context["visible_enemy_summary"],
                    ],
                    dim=-1,
                )
            )
            belief_policy_features = self.belief_policy_proj(
                torch.cat(
                    [belief_context, actor_enemy_context["unseen_enemy_summary"]],
                    dim=-1,
                )
            )
            actor_features = base_actor_features
        else:
            actor_enemy_features = actor_enemy_context["actor_enemy_features"]
            if len(actor_enemy_features) == 1:
                actor_enemy_features = actor_enemy_features[0]
            else:
                actor_enemy_features = torch.cat(actor_enemy_features, dim=-1)

            actor_features = self.policy_proj(
                torch.cat(
                    [belief_context, move_feat, self_feat, ally_feat, actor_enemy_features],
                    dim=-1,
                )
            )
            belief_policy_features = None
        critic_context = actor_enemy_context["critic_context"]
        if self.use_latent_belief:
            critic_context = critic_context + latent_aux["latent_context"]
        aux = {
            "belief_mu": belief_mu,
            "belief_logvar": belief_logvar,
            "belief_u": belief_u,
            "enemy_visible": current["enemy_visible"],
            "visible_enemy_summary": actor_enemy_context["visible_enemy_summary"],
            "unseen_enemy_summary": actor_enemy_context["unseen_enemy_summary"],
            "enemy_summary": actor_enemy_context["enemy_summary"],
            "critic_context": critic_context,
        }
        aux.update(latent_aux)
        if belief_policy_features is not None:
            aux["belief_policy_features"] = belief_policy_features
        return actor_features, new_memory, next_latent_state, next_history, aux

    def _encode(self, obs, prev_actions, agent_ids, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if prev_actions is not None:
            prev_actions = check(prev_actions).to(**self.tpdv)
        if agent_ids is not None:
            agent_ids = check(agent_ids).to(**self.tpdv)

        # Rollout path: one hidden state per sample.
        if rnn_states.size(0) == obs.size(0):
            memory, latent_state, history_tokens = self._unpack_actor_state(rnn_states)
            memory = memory * masks
            if latent_state is not None:
                latent_state = latent_state * masks
            if history_tokens is not None:
                history_tokens = history_tokens * masks.view(-1, 1, 1, 1)
            actor_features, new_memory, next_latent_state, next_history, aux = self._encode_step(
                obs,
                prev_actions,
                agent_ids,
                memory,
                latent_state,
                history_tokens,
            )
            return actor_features, self._pack_actor_state(new_memory, next_latent_state, next_history), aux

        # Training path: recurrent mini-batches provide initial memory for each chunk.
        batch_size = rnn_states.size(0)
        seq_len = obs.size(0) // batch_size
        obs = obs.view(seq_len, batch_size, -1)
        masks = masks.view(seq_len, batch_size, -1)
        if prev_actions is not None:
            prev_actions = prev_actions.view(seq_len, batch_size, -1)
        if agent_ids is not None:
            agent_ids = agent_ids.view(seq_len, batch_size, -1)

        memory, latent_state, history_tokens = self._unpack_actor_state(rnn_states)
        actor_features = []
        belief_mu = []
        belief_logvar = []
        belief_u = []
        enemy_visible = []
        visible_enemy_summary = []
        unseen_enemy_summary = []
        enemy_summary = []
        critic_context = []
        belief_policy_features = [] if self.use_split_belief_logits else None
        latent_prior_mu = [] if self.use_latent_belief else None
        latent_prior_logvar = [] if self.use_latent_belief else None
        latent_post_mu = [] if self.use_latent_belief else None
        latent_post_logvar = [] if self.use_latent_belief else None

        for t in range(seq_len):
            memory = memory * masks[t]
            if latent_state is not None:
                latent_state = latent_state * masks[t]
            if history_tokens is not None:
                history_tokens = history_tokens * masks[t].view(batch_size, 1, 1, 1)
            step_prev_actions = prev_actions[t] if prev_actions is not None else None
            step_agent_ids = agent_ids[t] if agent_ids is not None else None
            step_features, memory, latent_state, history_tokens, step_aux = self._encode_step(
                obs[t],
                step_prev_actions,
                step_agent_ids,
                memory,
                latent_state,
                history_tokens,
            )
            actor_features.append(step_features)
            belief_mu.append(step_aux["belief_mu"])
            belief_logvar.append(step_aux["belief_logvar"])
            belief_u.append(step_aux["belief_u"])
            enemy_visible.append(step_aux["enemy_visible"])
            visible_enemy_summary.append(step_aux["visible_enemy_summary"])
            unseen_enemy_summary.append(step_aux["unseen_enemy_summary"])
            enemy_summary.append(step_aux["enemy_summary"])
            critic_context.append(step_aux["critic_context"])
            if belief_policy_features is not None:
                belief_policy_features.append(step_aux["belief_policy_features"])
            if self.use_latent_belief:
                latent_prior_mu.append(step_aux["latent_prior_mu"])
                latent_prior_logvar.append(step_aux["latent_prior_logvar"])
                latent_post_mu.append(step_aux["latent_post_mu"])
                latent_post_logvar.append(step_aux["latent_post_logvar"])

        actor_features = torch.cat(actor_features, dim=0)
        aux = {
            "belief_mu": torch.cat(belief_mu, dim=0),
            "belief_logvar": torch.cat(belief_logvar, dim=0),
            "belief_u": torch.cat(belief_u, dim=0),
            "enemy_visible": torch.cat(enemy_visible, dim=0),
            "visible_enemy_summary": torch.cat(visible_enemy_summary, dim=0),
            "unseen_enemy_summary": torch.cat(unseen_enemy_summary, dim=0),
            "enemy_summary": torch.cat(enemy_summary, dim=0),
            "critic_context": torch.cat(critic_context, dim=0),
        }
        if belief_policy_features is not None:
            aux["belief_policy_features"] = torch.cat(belief_policy_features, dim=0)
        if self.use_latent_belief:
            aux["latent_prior_mu"] = torch.cat(latent_prior_mu, dim=0)
            aux["latent_prior_logvar"] = torch.cat(latent_prior_logvar, dim=0)
            aux["latent_post_mu"] = torch.cat(latent_post_mu, dim=0)
            aux["latent_post_logvar"] = torch.cat(latent_post_logvar, dim=0)
        return actor_features, self._pack_actor_state(memory, latent_state, history_tokens), aux

    def get_critic_context(self, obs, prev_actions, agent_ids, rnn_states, masks):
        _, _, aux = self._encode(obs, prev_actions, agent_ids, rnn_states, masks)
        return aux["critic_context"]

    def _build_action_distribution(self, actor_features, belief_policy_features, available_actions=None):
        if not self.use_split_belief_logits:
            return self.act.action_out(actor_features, available_actions)

        base_logits = self.base_action_out.linear(actor_features)
        # Keep MAPPO's base policy in charge of movement/control actions and let belief only bias attack targets.
        attack_bias = torch.tanh(self.belief_attack_out.linear(belief_policy_features))
        belief_logits = torch.zeros_like(base_logits)
        belief_logits[:, self.n_actions_no_attack:] = attack_bias
        policy_alpha = self._belief_policy_alpha.to(
            device=base_logits.device, dtype=base_logits.dtype
        )
        action_logits = base_logits + policy_alpha * belief_logits
        if available_actions is not None:
            action_logits = action_logits.masked_fill(available_actions == 0, -1e10)
        return FixedCategorical(logits=action_logits)

    def forward(
        self,
        obs,
        prev_actions,
        agent_ids,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
        return_aux=False,
    ):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, new_rnn_states, aux = self._encode(
            obs, prev_actions, agent_ids, rnn_states, masks
        )
        if self.use_split_belief_logits:
            action_dist = self._build_action_distribution(
                actor_features,
                aux["belief_policy_features"],
                available_actions,
            )
            actions = action_dist.mode() if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_probs(actions)
        else:
            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        if return_aux:
            return actions, action_log_probs, new_rnn_states, aux
        return actions, action_log_probs, new_rnn_states

    def evaluate_actions(
        self,
        obs,
        prev_actions,
        agent_ids,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        return_aux=False,
    ):
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, _, aux = self._encode(obs, prev_actions, agent_ids, rnn_states, masks)
        if self.use_split_belief_logits:
            action_dist = self._build_action_distribution(
                actor_features,
                aux["belief_policy_features"],
                available_actions,
            )
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None and self._use_policy_active_masks:
                dist_entropy = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy().mean()
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None,
            )
        if return_aux:
            return action_log_probs, dist_entropy, aux
        return action_log_probs, dist_entropy


class ResidualBeliefActor(BeliefTransformerActor):
    """Plain MAPPO backbone plus a bounded residual correction from the belief branch."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, action_space, device)
        if self.use_latent_belief:
            raise NotImplementedError("use_latent_belief is not yet supported with use_residual_belief_actor=True")
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) != 1:
            raise NotImplementedError("ResidualBeliefActor currently expects flat observations")

        self.base = MLPBase(args, obs_shape)
        self.base_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        self.belief_residual_alpha = getattr(args, "belief_residual_alpha", 0.3)
        self.belief_residual_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.belief_actor_state_size = self.actor_state_size
        self.actor_state_size = self.hidden_size + self.belief_actor_state_size
        self.policy_in_dim = self.hidden_size * 5
        self.policy_proj = nn.Sequential(
            nn.Linear(self.policy_in_dim, self.hidden_size),
            nn.ReLU(),
        )

        # Reinterpret the policy alpha schedule as the residual strength schedule for this actor.
        self.belief_policy_alpha = self.belief_residual_alpha
        if self.scheduled_belief_policy:
            self._belief_policy_alpha.fill_(0.0)
        else:
            self._belief_policy_alpha.fill_(float(self.belief_policy_alpha))

        self.to(device)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.actor_state_size)

    def _policy_belief_alpha_tensor(self, ref_tensor):
        # Residual mode extracts an unscaled belief branch, then applies the schedule at the residual output.
        return torch.ones(1, device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _encode_step(self, obs, prev_actions, agent_ids, memory, history_tokens):
        step_tokens, current = self.tokenizer(obs, prev_actions=prev_actions, agent_ids=agent_ids)
        history_seq, next_history = self._compose_history_tokens(step_tokens, history_tokens)
        history_emb = F.relu(self.token_embed(history_seq))
        new_memory = self._update_memory(history_emb, memory)

        belief_mu = self.belief_mu_head(new_memory).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        belief_logvar = self.belief_logvar_head(new_memory).view(-1, self.n_enemies, self.enemy_state_feat_dim)
        belief_u = self.belief_u_head(new_memory).view(
            -1, self.n_enemies, self.enemy_state_feat_dim, self.belief_lowrank_rank
        )

        move_feat = F.relu(self.token_embed(current["move_token"]))
        self_feat = F.relu(self.token_embed(current["self_token"]))
        ally_feat = self._masked_mean(
            F.relu(self.token_embed(current["ally_tokens"])),
            current["ally_visible"],
        )
        visible_enemy_feat = F.relu(self.token_embed(current["enemy_tokens"]))
        # Scheduled detach keeps belief self-supervised early, then gradually hands policy gradients back later in training.
        detach_alpha = self._belief_detach_alpha.to(device=belief_mu.device, dtype=belief_mu.dtype)
        belief_actor_src = belief_mu.detach() * detach_alpha + belief_mu * (1.0 - detach_alpha)
        # A separate forward alpha delays how strongly belief can steer action logits.
        belief_enemy_feat = F.relu(self.belief_enemy_proj(belief_actor_src))
        policy_alpha = self._policy_belief_alpha_tensor(belief_mu)
        policy_belief_enemy_feat = belief_enemy_feat * policy_alpha
        enemy_visible = current["enemy_visible"].float()

        unseen_belief_mask = self._build_unseen_belief_mask(enemy_visible, belief_logvar)
        fused_enemy_feat = (
            enemy_visible.unsqueeze(-1) * visible_enemy_feat
            + unseen_belief_mask.unsqueeze(-1) * policy_belief_enemy_feat
        )
        enemy_summary_mask = enemy_visible + unseen_belief_mask
        enemy_summary = self._masked_mean(fused_enemy_feat, enemy_summary_mask)

        critic_fused_enemy_feat = (
            enemy_visible.unsqueeze(-1) * visible_enemy_feat
            + unseen_belief_mask.unsqueeze(-1) * belief_enemy_feat
        )
        critic_context = self._masked_mean(critic_fused_enemy_feat, enemy_summary_mask)

        actor_features = self.policy_proj(
            torch.cat(
                [new_memory, move_feat, self_feat, ally_feat, enemy_summary],
                dim=-1,
            )
        )
        aux = {
            "belief_mu": belief_mu,
            "belief_logvar": belief_logvar,
            "belief_u": belief_u,
            "enemy_visible": current["enemy_visible"],
            "visible_enemy_summary": enemy_summary,
            "unseen_enemy_summary": enemy_summary,
            "enemy_summary": enemy_summary,
            "critic_context": critic_context,
        }
        return actor_features, new_memory, next_history, aux

    def _pack_actor_state(self, base_rnn_states, belief_memory):
        if belief_memory.dim() == 2:
            belief_memory = belief_memory.unsqueeze(1)
        if belief_memory.size(1) == 1 and self._recurrent_N > 1:
            belief_memory = belief_memory.expand(-1, self._recurrent_N, -1)
        return torch.cat([base_rnn_states, belief_memory], dim=-1)

    def _unpack_actor_state(self, rnn_states):
        rnn_states = check(rnn_states).to(**self.tpdv)
        if rnn_states.size(-1) == self.hidden_size:
            base_rnn_states = rnn_states
            belief_memory = torch.zeros(
                rnn_states.size(0), 1, self.belief_actor_state_size, dtype=rnn_states.dtype, device=rnn_states.device
            )
            return base_rnn_states, belief_memory

        if rnn_states.size(-1) != self.actor_state_size:
            raise ValueError(
                f"ResidualBeliefActor expected packed actor state size {self.actor_state_size}, got {rnn_states.size(-1)}"
            )

        base_rnn_states = rnn_states[..., :self.hidden_size].contiguous()
        belief_memory = rnn_states[..., self.hidden_size:].contiguous()
        return base_rnn_states, belief_memory

    def _encode_base(self, obs, base_rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        base_rnn_states = check(base_rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        base_actor_features = self.base(obs)
        base_actor_features, new_base_rnn_states = self.base_rnn(base_actor_features, base_rnn_states, masks)
        return base_actor_features, new_base_rnn_states

    def _encode(self, obs, prev_actions, agent_ids, rnn_states, masks):
        base_rnn_states, belief_memory = self._unpack_actor_state(rnn_states)
        base_actor_features, new_base_rnn_states = self._encode_base(obs, base_rnn_states, masks)
        belief_actor_features, new_belief_states, aux = super()._encode(
            obs, prev_actions, agent_ids, belief_memory, masks
        )

        residual_alpha = self._belief_policy_alpha.to(
            device=belief_actor_features.device, dtype=belief_actor_features.dtype
        )
        belief_delta = self.belief_residual_proj(belief_actor_features)
        actor_features = base_actor_features + residual_alpha * torch.tanh(belief_delta)
        new_rnn_states = self._pack_actor_state(new_base_rnn_states, new_belief_states)

        aux["belief_residual_norm"] = torch.norm(torch.tanh(belief_delta), dim=-1, keepdim=True)
        return actor_features, new_rnn_states, aux


BeliefCritic = BeliefAwareCritic
