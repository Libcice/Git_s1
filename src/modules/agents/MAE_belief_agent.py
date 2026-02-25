import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from components.masked_transformer_state import Base_Transformer as MT
from modules.layer.self_without_atten import Self_Without_Attention

class MAERNNAgent(nn.Module):
    def __init__(self, input_shape, args,mt_pretraining = False):
        self.mt_pretraining = mt_pretraining
        super(MAERNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if self.args.use_MT_mode and self.mt_pretraining is not True :
            # self.fc0 = nn.Linear(self.input_shape,int(args.rnn_hidden_dim/2))
            # 加上信念状态
            # self.fc1 = nn.Linear(self.input_shape+args.state_shape, int(args.rnn_hidden_dim/2))
            self.fc1 = nn.Linear(self.input_shape+args.state_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
            # statedim
            self.mae = MT(self.args.obs_shape, 1,args.state_shape, self.args,"cuda" if self.args.use_cuda else "cpu",
                          positional_type = self.args.positional_encoding_target)
            for param in self.mae.parameters() :
                param.requires_grad= False
            
            self.input_mae_obs_base = th.zeros((3000,self.args.n_agents*self.args.MT_traj_length,self.args.obs_shape)).to("cuda" if self.args.use_cuda else "cpu")
            self.input_mae_action_base = th.zeros((3000,self.args.n_agents*self.args.MT_traj_length,1)).to("cuda" if self.args.use_cuda else "cpu")
            
            #self.seq_num = th.arange(3000)
            #self.agent_location = th.arange(self.args.n_agents).repeat(1000)    
        else :            
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        

    def forward(self, inputs, all_info , hidden_state):
        
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)

        if self.args.use_MT_mode is not True or self.mt_pretraining :
            x = F.relu(self.fc1(inputs), inplace=True)
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hh = self.rnn(x, h_in)

            if getattr(self.args, "use_layer_norm", False):
                q = self.fc2(self.layer_norm(hh))
            else:
                q = self.fc2(hh)

            return q.view(b, a, -1), hh.view(b, a, -1)
        else :
            input_mae_obs, input_mae_action = self.generate_mae_input(all_info)
            mae_obs, mae_action, global_state_hat = self.mae(1, input_mae_obs, input_mae_action, train=False)
            # 取最后一帧全局状态(B, state_dim)
            global_vec = global_state_hat[:, -1, :]  
            # print("inputs.shape:", inputs.shape)
            # print("global_state_hat.shape:", global_state_hat.shape)
            # print("input_mae_obs.shape:", input_mae_obs.shape)
            # print("input_mae_action.shape:", input_mae_action.shape)
            # inputs.shape: torch.Size([6, 104])
            # global_state_hat.shape: torch.Size([6, 50, 146])
            # input_mae_obs.shape: torch.Size([6, 300, 84])

            # start = (self.args.MT_traj_length-1)*self.args.n_agents
            # mae_obs[:,start,:] = inputs
            # 把当前观测和全局向量拼在一起
            # x = F.relu(self.fc1(th.cat([inputs, global_vec.view(-1, global_vec.size(-1))], dim=1)), inplace=True)
            x = F.relu(self.fc1(th.cat([inputs, global_vec], dim=1)))
            h_in = hidden_state.reshape(-1,self.args.rnn_hidden_dim)
            h = self.rnn(x,h_in)
            if getattr(self.args, "use_layer_norm", False):
                q = self.fc2(self.layer_norm(h))
            else:
                q = self.fc2(h)

            return q.view(b, a, -1), h.view(b, a, -1), global_vec
            

    def generate_mae_input(self, all_info):
        obs_all, action_all = all_info[0].flatten(0, 1), all_info[1].flatten(0, 1)
        # 确保动作长度和观测一致
        if action_all.size(1) == obs_all.size(1) - 1:
            # 补最后一个动作
            last_action = action_all[:, -1:, :]
            action_all = th.cat([action_all, last_action], dim=1)
        elif action_all.size(1) != obs_all.size(1):
            raise RuntimeError(f"obs len={obs_all.size(1)}, action len={action_all.size(1)} mismatch!")

        # 1111obs_all.size2 torch.Size([6, 50, 84])
        # action_all.size22222 torch.Size([6, 50, 1])

        # print('1111obs_all.size2',obs_all.shape)
        # print('action_all.size22222',action_all.shape)  
        input_mae_obs = self.input_mae_obs_base[0:obs_all.shape[0], :, :].clone().detach()
        input_mae_action = self.input_mae_action_base[0:action_all.shape[0], :, :].clone().detach()

        input_mae_obs[:, ::self.args.n_agents, :] = obs_all[:, ::1, :]
        input_mae_action[:, ::self.args.n_agents, :] = action_all[:, ::1, :] / self.args.n_actions

        return input_mae_obs, input_mae_action  
            
            