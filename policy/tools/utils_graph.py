import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool

from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops, softmax
    


import math
import numpy as np

from termcolor import cprint

import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    
def padding_list(data, n_padding, if_float=True):
    """
    Function : 
        Fit action space with ACTION_NODE_SIZE padding for pointer net input (transformer-based)
        Fit state space with GRAPH_NODE_SIZE padding for transformer-based net input
    Input :
        data : Batch, Num_Action / Batch. Num_Node, Feature_Dim
        n_padding : ACTION_NODE_SIZE / GRAPH_NODE_SIZE
        if_float : using float data when padding feature, false when padding action
    Output :
        out_data : Batch, Num_Key(Padded) / Batch. Num_Key(Padded), Feature_Dim
        out_mask : Batch, Num_Key(Padded) (1, original, 0, padded)
    """
    list_data = []
    list_mask = []
    # Batch, n_key = > Batch, n_padding
    for idx in range(len(data)):
        data_with_idx = data[idx]
        
        assert len(data_with_idx) <= n_padding
        data_with_idx = torch.tensor(data_with_idx).unsqueeze(-1) if not if_float else torch.tensor(data_with_idx)
        mask_with_idx = torch.ones((data_with_idx.shape[0], 1))
        
        data_padding = torch.nn.ZeroPad2d((0, 0, 0, n_padding - len(data_with_idx))) # left, right. top, bottom
        data_padded = data_padding(data_with_idx)
        mask_padded = data_padding(mask_with_idx)
        
        list_data.append(data_padded.unsqueeze(0))
        list_mask.append(mask_padded.unsqueeze(0))
        
    out_data = torch.concat(list_data, dim=0)
    out_mask = torch.concat(list_mask, dim=0).squeeze(-1)
    
    out_data = out_data.float() if if_float else out_data.squeeze(-1)
    
    return out_data, out_mask

def padding_graph(node, batch, n_padding):
    """
    Function : 
        Fit GCN input with Transformer input
    Input :
        node : Num_Node, Dim_Feature
        batch : Num_Node, 
        n_padding : GRAPH_NODE_SIZE
    Output :
        out_node : Batch, Num_Key(Padded), Dim_Feature
        out_mask : Batch, Num_Query = 1, Num_Key(Padded) (1, original, 0, padded)
    """
    batch_size = max(batch)+1
    list_batch = []
    list_mask = []
    for idx in range(batch_size):
        node_with_idx = node[batch==idx]
        # print("batch:", batch) # [0,0,0,0,0] (len==num_per_batch)
        # print("idx:", idx) # 0
        # print("node_with_idx:", node_with_idx) # (num_per_batch, 13)
        # print("n_padding:", n_padding) # 1000
        # print("node_with_idx.shape[0]:", node_with_idx.shape[0]) # num_per_batch
        assert node_with_idx.shape[0] <= n_padding
        mask_with_idx = torch.ones((node_with_idx.shape[0],1)).cuda()
        node_padding = torch.nn.ZeroPad2d((0, 0, 0, n_padding - node_with_idx.shape[0])) # left, right. top, bottom
        node_padded = node_padding(node_with_idx)
        mask_padded = node_padding(mask_with_idx)
        
        list_batch.append(node_padded.unsqueeze(0))
        list_mask.append(mask_padded.squeeze(-1).unsqueeze(0))
    
    out_node = torch.concat(list_batch, dim=0)
    out_mask = torch.concat(list_mask, dim=0).unsqueeze(1)    
    
    return out_node, out_mask

# def padding_graph_v2(node, batch, edge_index):
def padding_graph_v2(node, batch, args):
    """
    Function : 
        Fit GCN input with Transformer input
    Input :
        node : Num_Node, Dim_Feature
        batch : Num_Node, 
    Output :
        out_node : Batch, Num_Key(Padded), Dim_Feature
        out_mask : Batch, Num_Query = 1, Num_Key(Padded) (1, original, 0, padded)
        edge_matrix : Batch, Num_Key(Padded), Num_Key(Padded)
    """
    batch_size = batch[-1] + 1
    
    list_batch = []
    list_mask = []

    max_padding = -1
    for idx in range(batch_size):
        node_with_idx = node[batch==idx]
        # print("node_with_idx.shape[0]:", node_with_idx.shape[0])
        max_padding = node_with_idx.shape[0] if node_with_idx.shape[0] > max_padding else max_padding
    n_padding = max_padding
    args.graph_num_graph_padding = n_padding
    # print("======================> batch: <===============", batch)
    # print("======================> n_padding: {} <=================".format(n_padding))
        
    for idx in range(batch_size):
        node_with_idx = node[batch==idx]

        assert node_with_idx.shape[0] <= n_padding
        mask_with_idx = torch.ones((node_with_idx.shape[0],1)).cuda()
        node_padding = torch.nn.ZeroPad2d((0, 0, 0, n_padding - node_with_idx.shape[0])) # left, right. top, bottom
        node_padded = node_padding(node_with_idx)
        mask_padded = node_padding(mask_with_idx)
        
        list_batch.append(node_padded.unsqueeze(0))
        list_mask.append(mask_padded.squeeze(-1).unsqueeze(0))
    
    out_node = torch.concat(list_batch, dim=0)
    out_mask = torch.concat(list_mask, dim=0).unsqueeze(1)    
    
    # edge_matrix = to_dense_adj(edge_index=edge_index, batch=batch, max_num_nodes=n_padding)
    
    # return out_node, out_mask, edge_matrix
    return out_node, out_mask



def to_adjacency_matrix(edge_adj, n_padding):
    """
    Function : 
        PyG edge adjacency => Adjacenct Matrix
    Input :
        edge_adj : Batch, 2, Num_Edge
        n_padding : GRAPH_NODE_SIZE
    Output :
        matrix : Batch, GRAPH_NODE_SIZE, GRAPH_NODE_SIZE
    """
    list_matrix = []
    for i in range(len(edge_adj)):
        matrix = torch.zeros((n_padding, n_padding))
        for j in range(len(edge_adj[i][0])):
            start = edge_adj[i][0][j]
            end = edge_adj[i][1][j]
            matrix[start][end] = 1

        list_matrix.append(matrix.unsqueeze(0))
    out_matrix = torch.concat(list_matrix, dim=0)
    return out_matrix

# ========================================> Node-Edge Encoder <========================================
class GATUnit(torch.nn.Module):
    """
    Function : 
        Graph Attention Network Unit
    Input : 
        state : Batch, Graph(PyG), Input_Dim with Edge (Num_Edge, Edge_dim)
        using_edge_attr : flag of using edge_attr or not
    Output : 
        x : Num_Node, Feature_Dim
    """
    def __init__(self, input_dim, output_dim, embedding_dim, edge_dim, n_head=2):
        super(GATUnit, self).__init__()
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.gat1 = GATv2Conv(input_dim, embedding_dim, heads=n_head, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(embedding_dim * n_head, output_dim, heads=1, edge_dim=edge_dim)
    
    def forward(self, x, edge_index, edge_attr, using_edge_attr=False):
        if using_edge_attr:
            x = self.gat1(x, edge_index, edge_attr=edge_attr)
            x = self.gat2(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.gat1(x, edge_index)
            x = self.gat2(x, edge_index)
        return x
    
    
class GATEnhanceEncoder(torch.nn.Module):
    """
    Function :
        Enhance node features with edge features and multi-hop neighbor node features
    Input : 
        state : Batch_size, Graph (PyG), Input_Dim with Edge (Num_Edge, Edge_dim)
    Output : 
        out : Num_Node. Feature_Dim (Batch)
    """
    def __init__(self, input_dim, edge_dim, output_dim, embedding_dim, n_head=2, n_layer=2, n_layer_with_edge_attr=10):
        super(GATEnhanceEncoder, self).__init__()
        self.n_layer_with_edge_attr = n_layer_with_edge_attr
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.pre = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList([GATUnit(input_dim=embedding_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, edge_dim=edge_dim, n_head=n_head) for i in range(n_layer)])
        self.linear_layers = torch_geometric.nn.Sequential(
            "x,", [
                # (nn.Dropout(p=0.5), "x -> xd"),
                (nn.Linear(embedding_dim, output_dim), "x -> x1"),
                (nn.LeakyReLU(), "x1 -> x_out"),
                ]
            ) 
    
    def forward(self, state):
        x, edge_index, edge_attr = state.x, state.edge_index, state.edge_attr
        # print("AAAAAAAAAAAAAAAAAA", edge_attr)
        x = F.leaky_relu(self.pre(x))
        for idx, layer in enumerate(self.layers):
            # edge_attr = torch.tensor(edge_attr, dtype=torch.int64)
            # print('--------edge_attr============:', edge_attr.dtype)
            if idx < self.n_layer_with_edge_attr:
                x = layer(x, edge_index, edge_attr, using_edge_attr=True)
            else:
                x = layer(x, edge_index, edge_attr, using_edge_attr=False)
        x = self.linear_layers(x)
        return x


# ========================================> Node-wise Encoder <========================================
class GCNUnit(torch.nn.Module):
    """
    Function : 
        GCN network unit
    Input : 
        state : Batch, Graph(PyG), Input_Feature_Dim
    Output : 
        x : Num_Node, Feature_Dim
    """
    def __init__(self, input_dim, output_dim, embedding_dim):
        super(GCNUnit, self).__init__()
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.model = torch_geometric.nn.Sequential(
            "x, edge_index, batch_index", [
                (GCNConv(input_dim, embedding_dim), "x, edge_index -> x1"),
                (nn.LeakyReLU(), "x1 -> x1a"),
                # (nn.Dropout(p=0.5), "x1a -> x1d"),
                (GCNConv(embedding_dim, output_dim), "x1a, edge_index -> x2"),
                (nn.LeakyReLU(), "x2 -> x_out"),
                ]
            ) 
    
    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index, batch)
        return x
    
    
class GCNEnhanceEncoder(torch.nn.Module):
    """
    Function :
        Enhance node features with only multi-hop neighbor node features
    Input : 
        state : Batch_size, Graph 
    Output : 
        out : Num_Node. Feature_Dim (Batch)
    """
    def __init__(self, input_dim, output_dim, embedding_dim, n_layer=1):
        super(GCNEnhanceEncoder, self).__init__()
        self.input_dim, self.output_dim, self.embedding_dim = input_dim, output_dim, embedding_dim
        self.pre = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [GCNUnit(input_dim=embedding_dim, output_dim=embedding_dim, embedding_dim=embedding_dim) for i in range(n_layer)]
            )
        self.linear_layers = torch_geometric.nn.Sequential(
            "x,", [
                # (nn.Dropout(p=0.5), "x -> xd"),
                (nn.Linear(embedding_dim, output_dim), "x -> x1"),
                (nn.LeakyReLU(), "x1 -> x_out"),
                ]
            ) 
            
    def forward(self, state):
        x, edge_index, batch = state.x, state.edge_index, state.batch
        x = F.leaky_relu(self.pre(x))
        for layer in self.layers:
            x = layer(x, edge_index, batch)
        x = self.linear_layers(x)
        return x
    

class SingleHeadAttention(nn.Module):
    """
    Function : 
        pointer network layer for policy output
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        k (key) : Batch, Num_Key, Dim_Feature
        mask : Batch, Num_Query, Num_Key
    Output : 
        attention : Batch, Num_Query, Num_Key
    Source : 
        @article{cao2023ariadne,
        title={Ariadne: A reinforcement learning approach using attention-based deep networks for exploration},
        author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
        journal={arXiv preprint arXiv:2301.11575},
        year={2023}
        }
    """
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()        
        self.tanh_coef = 100
        self.embedding_dim = embedding_dim
        self.norm_factor = 1 / math.sqrt(embedding_dim)

        self.w_query = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.w_key = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):
        
        n_query = q.size(1) # 1 in PointerNet Policy
        n_batch, n_key, n_dim = k.size()
        
        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q) # Batch, Num_Query, Dim_Feature
        K = torch.matmul(k_flat, self.w_key).view(shape_k) # Batch, Num_Key, Dim_Feature
        
        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2)) # Batch, Num_Query, Num_Key
        U = self.tanh_coef * torch.tanh(U)
        
        
        if mask is not None:
            mask = mask.unsqueeze(1) # Batch, Num_Query=1, Num_Key
            U = U.masked_fill(mask == 0, -1e8) # 0 means will not be selected as actions
        
        # cprint("Pointer Net : \n", color='green', attrs=['reverse', 'bold'])
        # print("U : ", U.detach().cpu().numpy())
        attention = torch.log_softmax(U, dim=-1) 
        # print("attention : ", attention.detach().cpu().numpy())
                
        return attention # Batch, Num_Query, Num_Key


class MultiHeadAttention(nn.Module):
    """
    Function : 
        standard multi head attention network for node feature enhancement
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        k (key) : Batch, Num_Key, Dim_Feature
        v (value) : Batch, Num_Value, Dim_Feature
        key_padding_mask : Batch, Num_Key,  (node padding mask)
        attn_mask : Batch, Num_Query, Num_Key (edge mask)
    Params : 
        embedding_dim : hidden layer length
        n_head : num of heads used in MHA
    Output : 
        out : Batch, Num_Query, Embedding_dim
        attention : Num_Heads, Batch, Num_Query, Num_Key
    """
    def __init__(self, embedding_dim, n_head=2):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        
        self.value_dim = self.embedding_dim // self.n_head
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_head, self.input_dim, self.key_dim))
        self.w_key   = nn.Parameter(torch.Tensor(self.n_head, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_head, self.input_dim, self.value_dim))
        self.w_out   = nn.Parameter(torch.Tensor(self.n_head, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        q_flat = q.contiguous().view(-1, n_dim)
        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        shape_q = (self.n_head, n_batch, n_query, -1)
        shape_k = (self.n_head, n_batch, n_key,   -1)
        shape_v = (self.n_head, n_batch, n_value, -1)
        
        Q = torch.matmul(q_flat, self.w_query).view(shape_q) # n_head, n_batch, n_query, key_dim
        K = torch.matmul(k_flat, self.w_key  ).view(shape_k) # n_head, n_batch, n_key,   key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v) # n_head, n_batch, n_value, value_dim
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_head, n_batch, n_query, n_key

        # ====== Mask ======
        if attn_mask is not None: # n_batch, n_query, n_key => n_head, n_batch, n_query, n_key
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None: # n_batch, n_key, 1 => n_head, n_batch, n_query, n_key
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask == 0, -1e8) # 0 means will not be considered

        attention = torch.softmax(U, dim=-1)  # n_head, n_batch, n_query, n_key
        heads = torch.matmul(attention, V)  # n_head, n_batch, n_query, value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_head * self.value_dim), # n_batch * n_query, n_head * value_dim(embedding_dim)
            self.w_out.view(-1, self.embedding_dim) # n_head * value_dim, embedding_dim
        ).view(-1, n_query, self.embedding_dim)# batch_size, n_query, embedding_dim

        return out, attention  


class LayerNormalization(nn.Module):
    """
    Function : 
        layer normalization on (Num_Query * Dim_Feature)
    Input : 
        input : Batch, Num_Query, Dim_Feature
    Output : 
        output : Batch, Num_Query, Dim_Feature
    """
    def __init__(self, embedding_dim):
        super(LayerNormalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    """
    Function : 
        Encoder Layer 
        using multi head attention network for node feature enhancement
        using layer norms & residual for data processing
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        key_padding_mask : Batch, Num_Key,  (node padding mask)
        attn_mask (edge connections) : Batch, Num_Query, Num_Key (edge mask)
    Params : 
        embedding_dim : hidden layer length
        n_head : num of heads used in MHA
    Output : 
        out : Batch, Num_Query, Embedding_dim
    """
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.ln1 = LayerNormalization(embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, 512), 
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512, embedding_dim))
        self.ln2 = LayerNormalization(embedding_dim)

    def forward(self, q, key_padding_mask=None, attn_mask=None):
        # MAH
        h0 = q
        h_normed_1 = self.ln1(q)
        h_enhanced_1, _ = self.multi_head_attention(q=h_normed_1, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h1 = h_enhanced_1 + h0
        
        # MLP
        h_normed_2 = self.ln2(h1)
        h_enhanced_2 = self.mlp(h_normed_2)
        h2 = h_enhanced_2 + h1
        return h2
    
    
class EncoderMultiHeadAttention(nn.Module):
    """
    Function : 
        Encoder Layer 
        using multi head attention network for node feature enhancement
        using layer norms & residual for data processing
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        key_padding_mask : Batch, Num_Key,  (node padding mask)
        attn_mask (edge connections) : Batch, Num_Query, Num_Key (edge mask)
    Params : 
        input_dim : input length
        embedding_dim : hidden layer length
        n_head : num of heads used in MHA
        n_layer : num of MHA layer
    Output : 
        out : Batch, Num_Query, Embedding_dim
    """
    
    def __init__(self, input_dim=4, embedding_dim=32, n_head=2, n_layer=1):
        super(EncoderMultiHeadAttention, self).__init__()
        self.init = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, q, key_padding_mask=None, attn_mask=None):
        q = self.init(q)
        for layer in self.layers:
            q = layer(q, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return q   
    
    
class DecoderLayer(nn.Module):
    """
    Function : 
        Decoder Layer : query the keys
        using multi head attention network for node feature enhancement
        using layer norms & residual for data processing
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        key_padding_mask : Batch, Num_Key,  (node padding mask)
        attn_mask (edge connections) : Batch, Num_Query, Num_Key (edge mask)
    Params : 
        embedding_dim : hidden layer length
        n_head : num of heads used in MHA
    Output : 
        out : Batch, Num_Query, Embedding_dim
        attention : Num_Heads, Batch, Num_Query, Num_Key
    """
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, n_head)
        self.ln1 = LayerNormalization(embedding_dim)
        self.ln2 = LayerNormalization(embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, 512),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512, embedding_dim))
        self.ln3 = LayerNormalization(embedding_dim)

    def forward(self, q, k, key_padding_mask=None, attn_mask=None):
        h0 = q
        h_normed_1 = self.ln1(q)
        k_normed = self.ln2(k)
        h_enhanced_1, attention = self.multi_head_attention(q=h_normed_1, k=k_normed, v=k_normed, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h1 = h_enhanced_1 + h0
        
        h_normed_2 = self.ln3(h1)
        h_enhanced_2 = self.mlp(h_normed_2)
        h2 = h_enhanced_2 + h1
        return h2, attention
    

class DecoderMultiHeadAttention(nn.Module):
    """
    Function : 
        Decoder Layer 
        using multi head attention network for node feature enhancement
        using layer norms & residual for data processing
    Input : 
        q (query) : Batch, Num_Query, Dim_Feature
        k (query) : Batch, Num_Key, Dim_Feature
        key_padding_mask : Batch, Num_Key,  (node padding mask)
        attn_mask (edge connections) : Batch, Num_Query, Num_Key (edge mask)
    Params : 
        input_dim : input length
        embedding_dim : hidden layer length
        n_head : num of heads used in MHA
        n_layer : num of MHA layer
    Output : 
        out : Batch, Num_Query, Embedding_dim
        attention : Num_Heads, Batch, Num_Query, Num_Key
    """
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(DecoderMultiHeadAttention, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, q, k, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            q, attention = layer(q, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return q, attention


class PointerNet(nn.Module):
    """
    Function:
        PointerNet as Policy Layer
    Input:
        k (key) : Batch, Num_Key, Dim_Feature
        q (query) : Batch, Num_Query, Dim_Feature
        mask : Batch, 1, Num_Key
    Output:
        logp_attention : Batch, Num_Key
    """
    def __init__(self, embedding_dim):
        super(PointerNet, self).__init__() 
        self.network = SingleHeadAttention(embedding_dim=embedding_dim)
        
    def forward(self, q, k, mask=None):
        return self.network(q, k, mask).squeeze(1)
    


ACTION_NODE_SIZE = 5 
GRAPH_NODE_SIZE = 7
if __name__ == '__main__':

    feature_dim = 4
    use_one = False
    if not use_one:
        node_info = [[[0,0,0,0], 
                    [1,2,3,4], 
                    [2,3,4,5]],
                    [[0,0,0,0], 
                    [3,4,5,6], 
                    [4,5,6,7],
                    [1,1,1,1]],
                    [[0,0,0,0], 
                    [5,6,7,8],
                    [6,7,8,9],
                    [7,8,9,0], 
                    [8,9,0,1]]]
        edge_info = [[[0,1,0,2],[1,0,2,0]],
                    [[0,1,0,2,0,3],[1,0,2,0,3,0]],
                    [[0,1,0,2,0,3,0,4],[1,0,2,0,3,0,4,0]]]
        edge_matrix = to_adjacency_matrix(edge_info, n_padding=GRAPH_NODE_SIZE)
        
        edge_features = [[[0,0,0,0,0],  
                          [1,1,1,1,1],
                          [2,2,2,2,2], 
                          [3,3,3,3,3]],
                         [[0,0,0,0,0], 
                          [1,1,1,1,1], 
                          [2,2,2,2,2],
                          [3,3,3,3,3], 
                          [0,0,0,0,0], 
                          [1,1,1,1,1]], 
                         [[0,0,0,0,0],  
                          [1,1,1,1,1], 
                          [2,2,2,2,2], 
                          [3,3,3,3,3],  
                          [0,0,0,0,0], 
                          [1,1,1,1,1], 
                          [0,0,0,0,0], 
                          [1,1,1,1,1]]]
        
        graph_batch = torch.tensor([0,0,0, 
                                    1,1,1,1, 
                                    2,2,2,2,2])  
        
        action_node_idx = [[[1]], 
                        [[1],[2]],
                        [[1],[2],[3]]]
        action_idx, action_mask = padding_list(action_node_idx, ACTION_NODE_SIZE, if_float=False) # XXX : Remember to delete current position
        
        current_idx = torch.tensor([0, 
                                    1, 
                                    2])
    else:
        node_info = [[[0,0,0,0], 
                      [1,2,3,4], 
                      [2,3,4,5]]]
        edge_info = [[[0,1,0,2],
                      [1,0,2,0]]]
        edge_features = torch.tensor(
            [[[0,0,0,0,0],
              [1,1,1,1,1],
              [2,2,2,2,2],
              [3,3,3,3,3]]]).float()

        edge_matrix = to_adjacency_matrix(edge_info, n_padding=GRAPH_NODE_SIZE)
        graph_batch = torch.tensor([0,0,0]) 
        action_node_idx = [[[1],[2]]]
        action_idx, action_mask = padding_list(action_node_idx, ACTION_NODE_SIZE, if_float=False)
        current_idx = torch.tensor([0])
    
    data_list = []
    for idx, data in enumerate(node_info):
        data_list.append(Data(x=torch.tensor(data).float(), edge_index=torch.tensor(edge_info[idx]).long(), edge_attr=torch.tensor(edge_features[idx]).float()))
    data_batch = Batch.from_data_list(data_list)
    # print("data_batch : \n", data_batch) 
    data_batch.validate(raise_on_error=True)
    
    if False:   
        # MultiHeadAttention
        # node_info_padded, node_padding_mask = padding_list(node_info, GRAPH_NODE_SIZE)
        # multi_head_attention =  MultiHeadAttention(embedding_dim=4, n_head=2)
        # out, attention = multi_head_attention(q=node_info_padded, key_padding_mask=node_padding_mask, attn_mask=edge_matrix)
        # print("out : ", out.shape) 
        # print("attention : ", attention.shape) 
        
        # EncoderLayer
        # node_info_padded, node_padding_mask = padding_list(node_info, GRAPH_NODE_SIZE)
        # encoder =  EncoderLayer(embedding_dim=4, n_head=2)
        # out = encoder(q=node_info_padded, key_padding_mask=node_padding_mask, attn_mask=edge_matrix)
        # print("out : \n", out.shape) 
        
        # EncoderMultiHeadAttention
        # node_info_padded, node_padding_mask = padding_list(node_info, GRAPH_NODE_SIZE)
        # encoder =  EncoderMultiHeadAttention(input_dim=4, embedding_dim=32, n_head=2, n_layer=3)
        # out = encoder(q=node_info_padded, key_padding_mask=node_padding_mask, attn_mask=edge_matrix)
        # print("out : \n", out.shape) 
        
        # DecoderLayer
        # node_info_padded, node_padding_mask = padding_list(node_info, GRAPH_NODE_SIZE)
        # decoder =  DecoderLayer(embedding_dim=4, n_head=2)
        # out, attention = decoder(q=node_info_padded, k=node_info_padded, key_padding_mask=node_padding_mask, attn_mask=edge_matrix)
        # print("out : \n", out.shape) 
        # print("attention : \n", attention.shape) 
        
        # DecoderMultiHeadAttention
        # node_info_padded, node_padding_mask = padding_list(node_info, GRAPH_NODE_SIZE)
        # decoder =  DecoderMultiHeadAttention(embedding_dim=4, n_head=2, n_layer=3)
        # out, attention = decoder(q=node_info_padded, k=node_info_padded, key_padding_mask=node_padding_mask, attn_mask=edge_matrix)
        # print("out : \n", out.shape) 
        # print("attention : \n", attention.shape) 
            
        # GATUnit
        # TODO : test cnn 
        gat = GATUnit(input_dim=4, output_dim=7, embedding_dim=32, edge_dim=5, n_head=2)
        out_gat = gat(data_batch.x, data_batch.edge_index, data_batch.edge_attr)
        # print("out_gat : ", out_gat.shape) 
        
        # GATEnhanceEncoder
        gatencoder = GATEnhanceEncoder(input_dim=4, output_dim=7, embedding_dim=32, edge_dim=5, n_head=2, n_layer=3, n_layer_with_edge_attr=2)
        out_gatencoder = gatencoder(data_batch)
        # print("out_gatencoder : ", out_gatencoder.shape) 
    
        # GCNUnit
        gcn_unit = GCNUnit(input_dim=4, output_dim=4, embedding_dim=32)
        out_gcn_unit = gcn_unit(data_batch.x, data_batch.edge_index, data_batch.batch)
        # print("out_gcn_unit : ", out_gcn_unit.shape) 
        
    # GCNEnhanceEncoder
    gcn = GCNEnhanceEncoder(input_dim=4, output_dim=4, embedding_dim=32, n_layer=2)
    out_gcn = gcn(data_batch)
    # print("out_gcn : ", out_gcn.shape) 