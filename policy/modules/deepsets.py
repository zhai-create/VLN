import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSet_Encoder(nn.Module):
    def __init__(self, args, dim_hidden=64): 
        super(DeepSet_Encoder, self).__init__()       
        dim_input = 4
        dim_output = 256
        self.num_agent = args.num_agent
        
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden, dim_output),
                nn.LeakyReLU())

    def forward(self, state):
        state = state.reshape(-1, self.num_agent - 1, 4)
        state = self.enc(state).sum(-2)
        state = self.dec(state)
        return state