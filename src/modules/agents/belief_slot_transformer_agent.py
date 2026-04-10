import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefSlotTransformerAgent(nn.Module):
    """Pure-transformer belief updater with one latent slot per enemy."""

    def __init__(self, input_shape, args):
        super(BeliefSlotTransformerAgent, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "belief_update_heads", getattr(args, "transformer_heads", 4))
        self.dropout = getattr(args, "belief_update_dropout", getattr(args, "transformer_dropout", 0.0))
        self.ff_mult = getattr(args, "belief_ff_mult", 4)
        self.n_enemies = args.enemy_num
        self.enemy_state_feat_dim = args.enemy_state_feat_dim
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)
        self.slot_aux_dim = 2

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by belief_update_heads")

        self.token_embed = nn.Linear(self.token_dim, self.hidden_dim)
        self.prev_action_embed = nn.Linear(args.n_actions, self.hidden_dim)

        self.init_belief_slots = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))

        self.memory_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.move_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.self_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.enemy_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.ally_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # Indexed belief needs a persistent identity for both observations and slots.
        self.enemy_index_embed = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))
        self.slot_index_embed = nn.Parameter(torch.zeros(1, self.n_enemies, self.hidden_dim))

        # The slot prior now sees two extra structural signals per enemy:
        # current visibility and how many consecutive steps it has remained unseen.
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + self.slot_aux_dim, self.hidden_dim * self.ff_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
        )
        self.prior_ln = nn.LayerNorm(self.hidden_dim)

        # Gate the carry-over path so slots cannot behave like a perfect cache.
        # The gate is later decayed as unseen_steps grows.
        self.prior_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 + self.slot_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # Legacy single-residual prior kept for rollback/reference.
        # prior_input = torch.cat([prev_belief_slots, memory_expand, action_expand], dim=-1)
        # prior_slots = self.prior_ln(prev_belief_slots + self.prior_mlp(prior_input))

        # Legacy two-MLP design kept for rollback/reference.
        # The current version uses a single prior MLP for slot propagation only,
        # and lets the transformer update memory from the posterior step.
        # self.memory_prior_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_dim * 3, self.hidden_dim * self.ff_mult),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
        # )
        # self.memory_prior_ln = nn.LayerNorm(self.hidden_dim)

        self.posterior_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.posterior_ln1 = nn.LayerNorm(self.hidden_dim)
        self.posterior_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.ff_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
        )
        self.posterior_ln2 = nn.LayerNorm(self.hidden_dim)

        self.latent_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.latent_ln1 = nn.LayerNorm(self.hidden_dim)
        self.latent_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.ff_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
        )
        self.latent_ln2 = nn.LayerNorm(self.hidden_dim)

        # When enemy i is visible, slot i should be explicitly corrected by the
        # matching enemy token instead of relying on unconstrained set attention.
        self.visible_slot_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.slot_aux_dim, self.hidden_dim * self.ff_mult),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
        )
        self.visible_slot_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.slot_aux_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.visible_slot_ln = nn.LayerNorm(self.hidden_dim)

        self.belief_mu_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, self.enemy_state_feat_dim)
        self.belief_u_head = nn.Linear(
            self.hidden_dim,
            self.enemy_state_feat_dim * self.belief_lowrank_rank,
        )

        # All enemies now go through posterior slots before entering Q.
        # We append visibility and unseen-time hints so the Q branch knows
        # whether a slot was strongly corrected by the current observation.
        self.slot_to_q = nn.Linear(self.hidden_dim + self.slot_aux_dim, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim * 6, args.n_actions)

    def init_hidden(self):
        return self.token_embed.weight.new_zeros(1, self.hidden_dim)

    def init_belief(self):
        return self.init_belief_slots

    def _masked_mean(self, feats, mask):
        if feats.size(1) == 0:
            return feats.new_zeros(feats.size(0), feats.size(-1))
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (feats * mask).sum(dim=1) / denom

    def _build_slot_aux(self, current_step):
        visible = current_step["enemy_visible"].float().unsqueeze(-1)
        # Compress unseen duration to a bounded feature while preserving order.
        unseen_feat = torch.tanh(current_step["unseen_steps"].float().unsqueeze(-1) / 5.0)
        return torch.cat([visible, unseen_feat], dim=-1)

    def _build_obs_sequence(self, current_step, prior_memory):
        move_feat = F.relu(self.token_embed(current_step["move_token"]))
        self_feat = F.relu(self.token_embed(current_step["self_token"]))
        enemy_feat = F.relu(self.token_embed(current_step["enemy_tokens"]))
        ally_feat = F.relu(self.token_embed(current_step["ally_tokens"]))

        memory_token = prior_memory.unsqueeze(1) + self.memory_type
        move_token = move_feat.unsqueeze(1) + self.move_type
        self_token = self_feat.unsqueeze(1) + self.self_type
        enemy_tokens = enemy_feat + self.enemy_type + self.enemy_index_embed
        ally_tokens = ally_feat + self.ally_type

        obs_seq = torch.cat([memory_token, move_token, self_token, enemy_tokens, ally_tokens], dim=1)
        bs_agents = obs_seq.size(0)
        fixed_valid = obs_seq.new_ones(bs_agents, 3)
        obs_valid = torch.cat([fixed_valid, current_step["enemy_visible"], current_step["ally_visible"]], dim=1)
        return obs_seq, obs_valid, move_feat, self_feat, enemy_feat, ally_feat

    def _update_latents(self, query_seq, obs_seq, obs_valid):
        key_padding_mask = obs_valid <= 0
        posterior_out, _ = self.posterior_attn(
            query=query_seq,
            key=obs_seq,
            value=obs_seq,
            key_padding_mask=key_padding_mask,
        )
        query_seq = self.posterior_ln1(query_seq + posterior_out)
        query_seq = self.posterior_ln2(query_seq + self.posterior_ff(query_seq))

        latent_out, _ = self.latent_attn(query=query_seq, key=query_seq, value=query_seq)
        query_seq = self.latent_ln1(query_seq + latent_out)
        query_seq = self.latent_ln2(query_seq + self.latent_ff(query_seq))
        return query_seq

    def _indexed_visible_update(self, belief_slots, enemy_feat, slot_aux, enemy_visible):
        indexed_enemy_feat = enemy_feat + self.enemy_index_embed
        visible_mask = enemy_visible.float().unsqueeze(-1)
        visible_input = torch.cat([belief_slots, indexed_enemy_feat, slot_aux], dim=-1)
        visible_candidate = self.visible_slot_fuse(visible_input)
        visible_gate = self.visible_slot_gate(visible_input)
        fused_visible_slots = self.visible_slot_ln(
            visible_gate * belief_slots + (1.0 - visible_gate) * visible_candidate
        )
        return visible_mask * fused_visible_slots + (1.0 - visible_mask) * belief_slots

    def forward(self, current_step, prev_belief_slots, prev_memory):
        prev_action_feat = F.relu(self.prev_action_embed(current_step["prev_action"]))
        memory_expand = prev_memory.unsqueeze(1).expand(-1, self.n_enemies, -1)
        action_expand = prev_action_feat.unsqueeze(1).expand(-1, self.n_enemies, -1)
        slot_aux = self._build_slot_aux(current_step)

        # New structural prior: each slot gets a learned candidate update and
        # a carry gate. The carry path is explicitly decayed when an enemy has
        # remained unseen for many steps, discouraging a perfect state cache.
        prior_input = torch.cat([prev_belief_slots, memory_expand, action_expand, slot_aux], dim=-1)
        prior_candidate = self.prior_mlp(prior_input)
        carry_gate = self.prior_gate(prior_input)
        unseen_decay = torch.exp(-0.25 * current_step["unseen_steps"].float().unsqueeze(-1).clamp(max=10.0))
        carry_gate = carry_gate * unseen_decay
        prior_slots = self.prior_ln(carry_gate * prev_belief_slots + (1.0 - carry_gate) * prior_candidate)

        # Legacy two-MLP memory prior kept for rollback/reference.
        # prior_belief_summary = prior_slots.mean(dim=1)
        # memory_prior_input = torch.cat([prev_memory, prior_belief_summary, prev_action_feat], dim=-1)
        # prior_memory = self.memory_prior_ln(prev_memory + self.memory_prior_mlp(memory_prior_input))

        # Keep memory as a carried-over latent and let cross-attention remain
        # the dominant memory updater.
        prior_memory = prev_memory

        obs_seq, obs_valid, move_feat, self_feat, enemy_feat, ally_feat = self._build_obs_sequence(current_step, prior_memory)

        # Indexed slot queries reduce permutation ambiguity: slot i now carries
        # a persistent slot identity before it attends to the enemy token set.
        slot_queries = prior_slots + self.slot_index_embed
        latent_queries = torch.cat([prior_memory.unsqueeze(1), slot_queries], dim=1)
        updated_latents = self._update_latents(latent_queries, obs_seq, obs_valid)

        new_memory = updated_latents[:, 0, :]
        belief_slots = updated_latents[:, 1:, :]

        # Directly anchor visible slots to the matching enemy token. This gives
        # slot i a privileged correction path from enemy token i when visible.
        belief_slots = self._indexed_visible_update(
            belief_slots,
            enemy_feat,
            slot_aux,
            current_step["enemy_visible"],
        )

        belief_mu = self.belief_mu_head(belief_slots)
        belief_logvar = self.belief_logvar_head(belief_slots)
        belief_u = self.belief_u_head(belief_slots).view(
            -1,
            self.n_enemies,
            self.enemy_state_feat_dim,
            self.belief_lowrank_rank,
        )

        ally_summary = self._masked_mean(ally_feat, current_step["ally_visible"])

        # Legacy Q path kept for reference: visible enemies bypassed slots.
        # visible_enemy_summary = self._masked_mean(enemy_feat, current_step["enemy_visible"])
        # hidden_enemy_summary = self._masked_mean(
        #     F.relu(self.slot_to_q(belief_slots)),
        #     1.0 - current_step["enemy_visible"].float(),
        # )

        # Indexed-slot Q path kept for reference: all enemies route through slots.
        # slot_q_feats = F.relu(self.slot_to_q(torch.cat([belief_slots, slot_aux], dim=-1)))
        # visible_enemy_summary = self._masked_mean(slot_q_feats, current_step["enemy_visible"])
        # hidden_enemy_summary = self._masked_mean(
        #     slot_q_feats,
        #     1.0 - current_step["enemy_visible"].float(),
        # )

        # Current hybrid Q path: keep directly observed enemy information on the
        # visible branch, and only substitute belief-derived slot features for
        # enemies that are currently hidden.
        visible_enemy_summary = self._masked_mean(enemy_feat, current_step["enemy_visible"])
        slot_q_feats = F.relu(self.slot_to_q(torch.cat([belief_slots, slot_aux], dim=-1)))
        hidden_enemy_summary = self._masked_mean(
            slot_q_feats,
            1.0 - current_step["enemy_visible"].float(),
        )

        q_input = torch.cat(
            [new_memory, move_feat, self_feat, ally_summary, visible_enemy_summary, hidden_enemy_summary],
            dim=-1,
        )
        q = self.q_head(q_input)

        return q, new_memory, belief_slots, belief_mu, belief_logvar, belief_u
