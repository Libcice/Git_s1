import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNBelief(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.mu_head  = nn.Linear(hidden_dim, state_dim)
        self.logsig_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, obs, h):
        h = self.gru(obs, h)
        mu = F.relu(self.mu_head(h))
        logsig = self.logsig_head(h)
        sigma = logsig.exp()
        return mu, sigma, h