import torch

from mappo_belief.smac_layout import build_smac_token_layout


def compute_belief_stats(args, share_obs, enemy_visible, belief_mu, belief_logvar, belief_u, active_masks):
    layout = build_smac_token_layout(args)
    belief_logvar = belief_logvar.clamp(
        min=getattr(args, "belief_logvar_min", -2.0),
        max=getattr(args, "belief_logvar_max", 1.0),
    )

    enemy_state = share_obs[:, layout.ally_state_dim:layout.ally_state_dim + layout.enemy_state_dim]
    enemy_state = enemy_state.view(-1, args.enemy_num, layout.enemy_state_feat_dim)
    unseen_mask = 1.0 - enemy_visible.float()
    alive_mask = (enemy_state[..., 0] > 0).float()
    agent_mask = active_masks.float().expand_as(unseen_mask)
    belief_mask = unseen_mask * alive_mask * agent_mask

    diff = enemy_state - belief_mu
    d_inv = torch.exp(-belief_logvar)
    du = d_inv.unsqueeze(-1) * belief_u
    a = torch.einsum("...dr,...ds->...rs", belief_u, du)
    rank = a.shape[-1]
    eye = torch.eye(rank, device=a.device, dtype=a.dtype).view(*([1] * (a.dim() - 2)), rank, rank)
    a = a + eye

    base_quad = (diff * d_inv * diff).sum(dim=-1)
    u_t_dinv_x = torch.einsum("...dr,...d->...r", belief_u, d_inv * diff)
    a_inv_u = torch.linalg.solve(a, u_t_dinv_x.unsqueeze(-1)).squeeze(-1)
    corr_quad = (u_t_dinv_x * a_inv_u).sum(dim=-1)
    quad = base_quad - corr_quad

    sign, logdet_a = torch.linalg.slogdet(a)
    if not torch.all(sign > 0):
        raise RuntimeError("Non-positive definite covariance factor encountered in belief NLL")
    logdet = belief_logvar.sum(dim=-1) + logdet_a
    raw_nll = 0.5 * (quad + logdet)
    nll = raw_nll / float(layout.enemy_state_feat_dim)
    nll = nll.clamp(min=0.0, max=getattr(args, "belief_nll_clip", 2.0))

    belief_loss_sum = (nll * belief_mask).sum()
    belief_denom = belief_mask.sum().clamp(min=1.0)
    belief_loss = belief_loss_sum / belief_denom
    raw_nll_mean = raw_nll.mean()
    belief_logvar_mean = belief_logvar.mean()
    belief_unseen_frac = belief_mask.mean()
    return {
        "belief_loss": belief_loss,
        "belief_raw_nll": raw_nll_mean,
        "belief_logvar_mean": belief_logvar_mean,
        "belief_unseen_frac": belief_unseen_frac,
    }


def compute_latent_kl_stats(args, post_mu, post_logvar, prior_mu, prior_logvar, active_masks):
    post_logvar = post_logvar.clamp(
        min=getattr(args, "latent_logvar_min", -4.0),
        max=getattr(args, "latent_logvar_max", 2.0),
    )
    prior_logvar = prior_logvar.clamp(
        min=getattr(args, "latent_logvar_min", -4.0),
        max=getattr(args, "latent_logvar_max", 2.0),
    )

    prior_var_inv = torch.exp(-prior_logvar)
    post_var = torch.exp(post_logvar)
    kl = 0.5 * (
        prior_logvar
        - post_logvar
        + (post_var + (post_mu - prior_mu) ** 2) * prior_var_inv
        - 1.0
    )
    kl = kl.mean(dim=-1, keepdim=True)

    mask = active_masks.float()
    if mask.dim() < kl.dim():
        mask = mask.expand_as(kl)
    kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1.0)

    return {
        "latent_kl_loss": kl_loss,
        "latent_kl_mean": kl.mean(),
        "latent_post_std_mean": torch.exp(0.5 * post_logvar).mean(),
        "latent_prior_std_mean": torch.exp(0.5 * prior_logvar).mean(),
    }
