import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Linear_Encoder(nn.Module):
    def __init__(self, args):
        super(Linear_Encoder, self).__init__()
        self.group_linear = nn.Sequential(
            nn.Linear(4 * (args.num_agent - 1), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU() # large network may fail
        )
    
    def forward(self, state):
        state = self.group_linear(state)        
        return state
    