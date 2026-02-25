import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.beliefs.rnn_belief import RNNBelief

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        # 信念网络
        self.belief_net = RNNBelief(
            input_dim=input_shape,
            hidden_dim=args.belief_hidden_dim,
            state_dim=args.state_shape   
        )

        # DRQN 网络 - 注意输入维度增加了信念维度
        total_input = input_shape + args.state_shape
        self.fc1 = nn.Linear(total_input, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # 返回双隐藏状态，每个都是2D
        h_drqn = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        h_belief = self.fc1.weight.new(1, self.args.belief_hidden_dim).zero_()
        if self.args.use_cuda:
            h_drqn = h_drqn.cuda()
            h_belief = h_belief.cuda()
        return (h_drqn, h_belief)

    def forward(self, inputs, hidden_state):
        h_drqn, h_belief = hidden_state

        # === 维度信息 ===
        b, n, _ = h_belief.size()                      # (B,N,H)
        #inputs.size()) 8 ，150
        # === belief 网络：3D → 2D → GRU → 2D → 3D ===
        h_belief_2d = h_belief.reshape(b * n, -1)      # (B*N,H)
        mu, sigma, h_belief_2d_new = self.belief_net(inputs, h_belief_2d)
        # belief_2d, h_belief_2d_new = self.belief_net(inputs, h_belief_2d)  # (B*N,D)
        # belief_2d 8，12  h_belief_2d_new(8 ,32) h_belief_new(1,8,32)
        h_belief_new = h_belief_2d_new.reshape(b, n, -1)  # (B,N,H)

        # === DRQN：2D 拼接 ===
        x = th.cat([inputs, mu], dim=-1)        # (B*N, D+E)
        x = F.relu(self.fc1(x))
        h_drqn_2d = h_drqn.reshape(b * n, -1)          # (B*N,H)
        h_drqn_new_2d = self.rnn(x, h_drqn_2d)         # (B*N,H)
        h_drqn_new = h_drqn_new_2d.reshape(b, n, -1)   # (B,N,H)
        q = self.fc2(h_drqn_new_2d).reshape(b, n, -1)  # (B,N,A)
        #输出信念
        return q, (h_drqn_new, h_belief_new), mu, sigma
