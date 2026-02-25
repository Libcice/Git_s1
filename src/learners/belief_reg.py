import torch as th
import torch.nn as nn
import torch.nn.functional as F

class BeliefRegLoss(nn.Module):
    def __init__(self, k, input_dim, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.k = k
        self.pred_future = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)   # ← 输出维 = 观测维
        )
        self.pred_past = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)   # ← 同上
        )
        self.pred_action = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)   # ← 输出维 = 动作维
        )

    def forward(self, belief_seq, obs_seq, act_seq, mask):
        # ---- 统一设备 ----
        device = next(self.parameters()).device      # 拿到网络所在设备
        belief_seq = belief_seq.to(device)
        obs_seq    = obs_seq.to(device)
        act_seq    = act_seq.to(device)
        mask       = mask.to(device)

        T, B, N, _ = belief_seq.shape
        loss_f = loss_i = loss_a = 0.
        valid = 0
        for t in range(T - self.k):
            b_t = belief_seq[t]
            if t + self.k < T:
                target_o = obs_seq[t + self.k]
                pred_o   = self.pred_future(b_t)
                loss_f  += F.mse_loss(pred_o, target_o, reduction='none').mean(-1)
                valid   += mask[t + self.k]
            if t >= self.k:
                target_o = obs_seq[t - self.k]
                pred_o   = self.pred_past(b_t)
                loss_i  += F.mse_loss(pred_o, target_o, reduction='none').mean(-1)
                valid   += mask[t - self.k]
            if t + 1 < T:
                target_a = act_seq[t + 1:t + 1 + self.k].mean(0)
                pred_a   = self.pred_action(b_t)
                loss_a  += F.mse_loss(pred_a, target_a, reduction='none').mean(-1)
                valid   += mask[t + 1:t + 1 + self.k].mean(0)
        return (loss_f + loss_i + loss_a) / valid.clamp(min=1), valid.sum()