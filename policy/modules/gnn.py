import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops, softmax
    
import math
import numpy as np


# Batch_size, Graph => Batch_size, Feature_dim
class GNN_Encoder(torch.nn.Module):
    def __init__(self, args):
        super(GNN_Encoder, self).__init__()
        self.num_agent = args.num_agent
        
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 256)
    
    def forward(self, state):
        state = state.reshape(-1, self.num_agent - 1, 4)
        ego = torch.zeros(state.shape[0], 1, 4).cuda()
        state = torch.cat([ego, state], dim=1) 

        data_list = []
        for _, data in enumerate(state):
            edge = torch.tensor([
                [0, 1, 0, 2, 0, 3],
                [1, 0, 2, 0, 3, 0]
            ]).cuda()
            data_list.append(Data(x=data, edge_index=edge))
        data_batch = Batch.from_data_list(data_list)
                
        x, edge_index = data_batch.x, data_batch.edge_index
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.fc(x))
        x = global_mean_pool(x, data_batch.batch)
        
        return x
    
      