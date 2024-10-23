import torch
import torch.nn as nn
import torch.nn.functional as F    
    
class LSTM_Encoder(nn.Module):
    def __init__(self, args):
        super(LSTM_Encoder, self).__init__()
        self.num_agent = args.num_agent
        self.sample_length = args.sample_length
        self.LSTM = nn.LSTM(
            input_size = 2 * (self.num_agent - 1),
            hidden_size = 256,        
            num_layers = 2,       
            batch_first = True,       #  (batch, time_step, input_size)
        )
        self.group_linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
    
    def forward(self, state):
        self.LSTM.flatten_parameters()
        state = state.reshape(-1, self.sample_length, 2 * (self.num_agent - 1))
        LSTM_out, _ = self.LSTM(state) # B, T, H
        state = self.group_linear(LSTM_out[:, -1, :])        
        
        return state