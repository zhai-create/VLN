import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.modules import fc, cnn, gnn, lstm, deepsets, set_transformer
from policy.tools.utils import weights_init

####################################
#             Network utils
####################################
class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.dim_feature = 256 + 64 + 64
        self.dim_embedding = 64
        
        modes = ['fc', 'cnn', 'gnn_vanilla', 'lstm', 'deepsets', 'set_transformer']
        net_modules = [fc.Linear_Encoder, cnn.Image_Encoder, gnn.GNN_Encoder, lstm.LSTM_Encoder, deepsets.DeepSet_Encoder, set_transformer.SetTransformer]
            
        mode_idx = modes.index(args.encoder_type)
        self.surrounding_embedding = net_modules[mode_idx](args)

        self.ego_embedding = nn.Linear(2, self.dim_embedding)
        self.goal_embedding = nn.Linear(2, self.dim_embedding)
        self.apply(weights_init)

    def forward(self, state):
        ego_state = F.leaky_relu(self.ego_embedding(state[0]))
        goal_state = F.leaky_relu(self.goal_embedding(state[1]))
        surrounding_state = self.surrounding_embedding(state[2])
    
        return torch.cat([ego_state, goal_state, surrounding_state], dim=1)
                
class FeatureMapper(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(FeatureMapper, self).__init__()
        
        self.fm = nn.Sequential(
            nn.Linear(dim_input, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)
    
    
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
        
def save_models(net, optimizer, name, dir_path):
    torch.save(net.state_dict(), dir_path + "_" + name)
    torch.save(optimizer.state_dict(), dir_path + "_" + name + "_optimizer")
        