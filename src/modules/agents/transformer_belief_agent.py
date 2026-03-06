import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBeliefAgent(nn.Module):
    """Transformer agent with a Gaussian belief head.

    Belief covariance is parameterized as:
        Sigma = diag(exp(logvar)) + U U^T
    where U has low rank.
    """

    def __init__(self, input_shape, args):
        super(TransformerBeliefAgent, self).__init__()
        self.args = args

        self.hidden_dim = getattr(args, "transformer_hidden_dim", args.rnn_hidden_dim)
        self.n_heads = getattr(args, "transformer_heads", 4)
        self.n_layers = getattr(args, "transformer_layers", 2)
        self.dropout = getattr(args, "transformer_dropout", 0.0)
        self.belief_lowrank_rank = getattr(args, "belief_lowrank_rank", 4)

        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads")

        self.obs_embed = nn.Linear(input_shape, self.hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)

        # Gaussian belief parameters.
        self.belief_mu_head = nn.Linear(self.hidden_dim, args.state_shape)
        self.belief_logvar_head = nn.Linear(self.hidden_dim, args.state_shape)
        self.belief_u_head = nn.Linear(self.hidden_dim, args.state_shape * self.belief_lowrank_rank)

        # Q uses fused obs and belief features.
        self.belief_feat = nn.Linear(args.state_shape, self.hidden_dim)
        self.q_head = nn.Linear(self.hidden_dim * 2, args.n_actions)

    def init_hidden(self):
        return self.obs_embed.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        # inputs: [batch*n_agents, input_shape]
        # hidden_state: [batch*n_agents, hidden_dim]
        obs_feat = F.relu(self.obs_embed(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        tokens = torch.stack([obs_feat, h_in], dim=1)
        encoded = self.encoder(tokens)

        h = encoded[:, 1, :]
        belief_mu = self.belief_mu_head(h)
        belief_logvar = self.belief_logvar_head(h)
        belief_u = self.belief_u_head(h).view(-1, self.args.state_shape, self.belief_lowrank_rank)

        belief_feat = F.relu(self.belief_feat(belief_mu))
        q_input = torch.cat([obs_feat, belief_feat], dim=-1)
        q = self.q_head(q_input)

        return q, h, belief_mu, belief_logvar, belief_u
