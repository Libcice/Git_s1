import copy
from components.belief_episode_buffer import EpisodeBatch_belief
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F   # ← 补上这一行
from torch.optim import RMSprop
# from learners.belief_reg import BeliefRegLoss


def gaussian_nll(mu, sigma, target):
    return F.gaussian_nll_loss(mu, target, sigma ** 2, reduction='none')


class QLearner_belief:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        # 再加正则头（如果启用）
        if getattr(self.args, 'use_belief_reg', False):
            # self.belief_reg = BeliefRegLoss(
            #     k=getattr(self.args, 'belief_reg_k', 5),
            #     input_dim=args.belief_dim,
            #     obs_dim=args.obs_shape,          # ← 观测维
            #     act_dim=args.n_actions,          # ← 动作维
            #     hidden_dim=getattr(args, 'belief_hidden_dim', 64),
            # )
            # self.params += list(self.belief_reg.parameters())   # ← 关键一行
            self.reg_coef = getattr(self.args, 'belief_reg_coef', 0.02)
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

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1


    def train(self, batch: EpisodeBatch_belief, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        N = getattr(self.args, "n_step", 1)
        if N == 1:
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        else:
            # N step Q-Learning targets
            n_rewards = th.zeros_like(rewards)
            gamma_tensor = th.tensor([self.args.gamma**i for i in range(N)], dtype=th.float, device=n_rewards.device)
            steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(N).long()
            for i in range(batch.max_seq_length - 1):
                n_rewards[:, i, 0] = ((rewards * mask)[:,i:i+N,0] * gamma_tensor[:(batch.max_seq_length - 1 - i)]).sum(dim=1)
            indices = th.linspace(0, batch.max_seq_length-2, steps=batch.max_seq_length-1, device=steps.device).unsqueeze(1).long()
            n_targets_terminated = th.gather(target_max_qvals*(1-terminated),dim=1,index=steps.long()+indices-1)
            targets = n_rewards + th.pow(self.args.gamma, steps.float()) * n_targets_terminated

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        #  新加正则化
        # === 2. 信念正则（Gaussian-NLL） ===
        if getattr(self.args, 'use_belief_reg', False):
            # 取 belief 分布参数  μ, σ
            # 注意：mac.forward 现在返回  agent_outs, hidden, mu, sigma
            mu_seq   = batch["belief"][:, :-1]          # [B, T-1, N, belief_dim]
            sigma_seq= batch["belief_sigma"][:, :-1]    # [B, T-1, N, belief_dim]
            state_target = batch["state"][:, :-1].unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)  # [B, T-1, 1, state_dim] → [B, T-1, N, state_dim]

            # reshape 到 2D
            B, T, N, _ = mu_seq.shape
            mu   = mu_seq.reshape(B*T*N, -1)
            sigma= sigma_seq.reshape(B*T*N, -1)
            target = state_target.reshape(B*T*N, -1)

            # Gaussian-NLL
            reg_loss = gaussian_nll(mu, sigma, target).mean(-1)   # 平均到特征维
            reg_loss = reg_loss.reshape(B, T, N)       

            mask_reg = batch["filled"][:, :-1].float().expand_as(reg_loss)   
            reg_loss = reg_loss * mask_reg          # 再乘 mask

            # print("TD loss:", loss.item())  0.18 → 0.016
            # print("Belief NLL:", reg_loss.sum().item() / mask_reg.sum().item())0.109 ± 0.001
            loss = loss + self.reg_coef * (reg_loss.sum() / mask_reg.sum())

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # # backward 之后、step 之前
        # grad_norm_belief = 0
        # for p in self.mac.agent.belief_net.parameters():
        #     if p.grad is not None:
        #         grad_norm_belief += p.grad.data.norm(2).item() ** 2
        # print("belief_net grad norm:", grad_norm_belief ** 0.5)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            agent_utils = (th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3) * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("agent_utils", agent_utils, t_env)
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
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
