import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops, softmax

from policy.tools.utils_graph import GCNEnhanceEncoder, GATEnhanceEncoder, EncoderMultiHeadAttention, DecoderMultiHeadAttention, PointerNet
from policy.tools.utils_graph import padding_list, padding_graph, padding_graph_v2, to_adjacency_matrix

import numpy as np

# ========================================> Graph + Pointer <========================================
class GraphPointerPolicy(nn.Module):
    """
    Function:
        Graph + PointerNet : 
            Using GCN / GAT / Transformer as graph encoder, PointerNet for node selection
            GCN / Transformer for only node feature settings
            GAT for both node feature and edge feature settings
    Input:
        using pyg:
            graphs : Batch, Graphs
        not using pyg:
            nodes (key) : Batch, Num_Node, Dim_Feature
            node_padding_mask : Batch, Num_Key
            edge_matrix : Batch, Num_Node, Num_Node
        
        current_idx: Batch, 1
        action_idx : Batch, Num_Action
        action_mask : Batch, Num_Action
    Output:
        logp_attention : Batch, Num_Key
    """
    def __init__(self, node_dim, edge_dim, embedding_dim, 
                 num_graph_padding,  
                 n_layer_encoder=2, n_layer_decoder=1,  
                 n_head=2, n_layer_with_edge_attr=1, 
                 encoder_type='GCN'):
        super(GraphPointerPolicy, self).__init__() 
        
        # Encoder
        self._encoder_type = encoder_type
        self._embedding_dim = embedding_dim
        if encoder_type == 'GCN':
            self.pre = GCNEnhanceEncoder(input_dim=node_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, 
                                         n_layer=n_layer_encoder)
        elif encoder_type == 'GAT':
            self.pre = GATEnhanceEncoder(input_dim=node_dim, output_dim=embedding_dim, embedding_dim=embedding_dim,
                                         edge_dim=edge_dim, n_head=n_head, 
                                         n_layer=n_layer_encoder, n_layer_with_edge_attr=n_layer_with_edge_attr)
        elif encoder_type == 'Transformer':
            self.pre = EncoderMultiHeadAttention(input_dim=node_dim, embedding_dim=embedding_dim, 
                                                 n_head=n_head, 
                                                 n_layer=n_layer_encoder)
        else:
            return NotImplementedError
        
        # Decoder
        self.decoder = DecoderMultiHeadAttention(embedding_dim=embedding_dim, n_head=n_head, n_layer=n_layer_decoder)
        self.pointer = PointerNet(embedding_dim=embedding_dim)
        self.combine_residual = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self._num_graph_padding = num_graph_padding
        
    def forward(self, state, args):
        # Graph Encoder for Data Enhancement
        if self._encoder_type == 'GCN' or self._encoder_type == 'GAT':
            graphs, current_idx, action_idx, action_mask = state
            node_enhanced = self.pre(graphs) # Batch, Graph => Num_Key, Feature_Dim
            # node_enhanced_padded, node_padding_mask = padding_graph(node_enhanced, graphs.batch, n_padding=self._num_graph_padding) # Batch, Num_Key(Padded), Dim_Feature & Batch, Num_Query=1, Num_Key
            node_enhanced_padded, node_padding_mask = padding_graph_v2(node_enhanced, graphs.batch, args)
        elif self._encoder_type == 'Transformer': 
            nodes, node_padding_mask, edge_matrix, current_idx, action_idx, action_mask = state
            node_enhanced_padded = self.pre(q=nodes, key_padding_mask=node_padding_mask, attn_mask=edge_matrix) # Batch, Num_Key, Feature_Dim
        else:
            NotImplementedError
        
        # Current Node 
        current_idx = current_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim) # Batch, 1, Feature_Dim
        current_node = torch.gather(node_enhanced_padded, 1, current_idx) # Batch, 1, Feature_Dim

        # Action Nodes
        action_idx = action_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim) # Batch, Num_Action, Feature_Dim

        # node_enhanced_padded: [1, 13, 64]
        # action_idx: [1, 100, 64]

        # ====================================================================
        
        action_node = torch.gather(node_enhanced_padded, 1, action_idx) # Batch, Num_Action, Feature_Dim
        
        # Current Node Feature Enhancement with Residual
        # enhanced_current_node, _ = self.decoder(current_node, action_node, action_mask) # enhance with action nodes
        enhanced_current_node, _ = self.decoder(current_node, node_enhanced_padded, node_padding_mask) # enhance with all nodes (Batch, 1, Embedding_dim)
        enhanced_current_node = self.combine_residual(torch.cat((enhanced_current_node, current_node), dim=-1)) # double the dim of embedding_dim (Batch, 1, 2*Embedding_dim)-->(Batch, 1, Embedding_dim)
        # PointerNet
        pointer_out = self.pointer(enhanced_current_node, action_node, action_mask) # Batch, Num_Action
        return pointer_out # Batch, Num_Action


class GraphQNet(nn.Module):
    """
    Function:
        Graph + QNet : 
            Using GCN / GAT / Transformer as graph encoder, Multi-Head-Attention for Q-Value
    Input: 
        
        using pyg:
            graphs : Batch, Graphs
        not using pyg:
            nodes (key) : Batch, Num_Node, Dim_Feature
            node_padding_mask : Batch, Num_Key
            edge_matrix : Batch, Num_Node, Num_Node
        
        current_idx: Batch, 1
        action_idx : Batch, Num_Action
        action_mask : Batch, Num_Action

    Output:
        q_values_masked : Batch, Num_Action, 1
        attention : Num_Heads, Batch, Num_Query=1, Num_Key
    """
    def __init__(self, node_dim, edge_dim, embedding_dim,
                 num_graph_padding, 
                 n_layer_encoder=4, n_layer_decoder=2,  
                 n_head=2, n_layer_with_edge_attr=2, 
                 encoder_type='GCN'):
        super(GraphQNet, self).__init__() 
        
        # Encoder
        self._encoder_type = encoder_type
        self._embedding_dim = embedding_dim
        self._num_graph_padding = num_graph_padding
        if encoder_type == 'GCN':
            self.pre = GCNEnhanceEncoder(input_dim=node_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, 
                                         n_layer=n_layer_encoder)
        elif encoder_type == 'GAT':
            self.pre = GATEnhanceEncoder(input_dim=node_dim, output_dim=embedding_dim, embedding_dim=embedding_dim,
                                         edge_dim=edge_dim, n_head=n_head, 
                                         n_layer=n_layer_encoder, n_layer_with_edge_attr=n_layer_with_edge_attr)
        elif encoder_type == 'Transformer':
            self.pre = EncoderMultiHeadAttention(input_dim=node_dim, embedding_dim=embedding_dim, 
                                                 n_head=n_head, 
                                                 n_layer=n_layer_encoder)
        else:
            return NotImplementedError
        
        # Decoder
        self.decoder_1 = DecoderMultiHeadAttention(embedding_dim=embedding_dim, n_head=n_head, n_layer=n_layer_decoder)
        self.action_enhancing_1 = nn.Linear(embedding_dim*3, embedding_dim)
        self.q_value_embedding_1 = nn.Linear(embedding_dim, 1)
        
        self.decoder_2 = DecoderMultiHeadAttention(embedding_dim=embedding_dim, n_head=n_head, n_layer=n_layer_decoder)
        self.action_enhancing_2 = nn.Linear(embedding_dim*3, embedding_dim)
        self.q_value_embedding_2 = nn.Linear(embedding_dim, 1)
        
        
    def forward(self, state, args):
        # Graph Encoder for Data Enhancement
        if self._encoder_type == 'GCN' or self._encoder_type == 'GAT':
            graphs, current_idx, action_idx, action_mask = state
            node_enhanced = self.pre(graphs) # Batch, Graph => Num_Key, Feature_Dim
            # node_enhanced_padded, node_padding_mask = padding_graph(node_enhanced, graphs.batch, n_padding=self._num_graph_padding) # Batch, Num_Key(Padded), Dim_Feature & Batch, Num_Query=1, Num_Key
            node_enhanced_padded, node_padding_mask = padding_graph_v2(node_enhanced, graphs.batch, args)
        elif self._encoder_type == 'Transformer': 
            nodes, node_padding_mask, edge_matrix, current_idx, action_idx, action_mask = state
            node_enhanced_padded = self.pre(q=nodes, key_padding_mask=node_padding_mask, attn_mask=edge_matrix) # Batch, Num_Key, Feature_Dim
        else:
            NotImplementedError
        
        # TODO : Augment Node Feature with Goal?
        
        current_idx = current_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim) # Batch, 1, Feature_Dim
        current_node = torch.gather(node_enhanced_padded, 1, current_idx) # Batch, 1, Feature_Dim

        action_idx = action_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim)
        action_node = torch.gather(node_enhanced_padded, 1, action_idx) # Batch, Num_Action, Feature_Dim
        
        # Q1
        # enhanced_current_node_1, attention_1 = self.decoder_1(current_node, action_node, action_mask) # enhance with action nodes
        enhanced_current_node_1, attention_1 = self.decoder_1(current_node, node_enhanced_padded, node_padding_mask) # enhance with all nodes
        enhanced_action_node_1 = torch.cat((enhanced_current_node_1.repeat(1, action_node.shape[1], 1), current_node.repeat(1, action_node.shape[1], 1), action_node), dim=-1)
        enhanced_action_node_1 = self.action_enhancing_1(enhanced_action_node_1) # (Batch, 3*Num_Action, Feature_Dim) --> (Batch, Num_Action, Feature_Dim)
        q_values_1 = self.q_value_embedding_1(enhanced_action_node_1) # Batch, Num_Action, 1
        zero_1 = torch.zeros_like(q_values_1).cuda()
        q_values_masked_1 = torch.where(action_mask.unsqueeze(-1) == 0, zero_1, q_values_1) # Batch, Num_Action, 1

        # Q2
        # enhanced_current_node_2, attention_2 = self.decoder_2(current_node, action_node, action_mask) # enhance with action nodes
        enhanced_current_node_2, attention_2 = self.decoder_2(current_node, node_enhanced_padded, node_padding_mask) # enhance with all nodes
        enhanced_action_node_2 = torch.cat((enhanced_current_node_2.repeat(1, action_node.shape[1], 1), current_node.repeat(1, action_node.shape[1], 1), action_node), dim=-1)
        enhanced_action_node_2 = self.action_enhancing_2(enhanced_action_node_2)
        q_values_2 = self.q_value_embedding_2(enhanced_action_node_2)
        zero_2 = torch.zeros_like(q_values_2).cuda()
        q_values_masked_2 = torch.where(action_mask.unsqueeze(-1) == 0, zero_2, q_values_2) # Batch, Num_Action, 1

        return q_values_masked_1, q_values_masked_2 

    def Q1(self, state, args):
        # Graph Encoder for Data Enhancement
        if self._encoder_type == 'GCN' or self._encoder_type == 'GAT':
            graphs, current_idx, action_idx, action_mask = state
            node_enhanced = self.pre(graphs) # Batch, Graph => Num_Key, Feature_Dim
            # node_enhanced_padded, node_padding_mask = padding_graph(node_enhanced, graphs.batch, n_padding=self._num_graph_padding) # Batch, Num_Key(Padded), Dim_Feature & Batch, Num_Query=1, Num_Key
            node_enhanced_padded, node_padding_mask = padding_graph_v2(node_enhanced, graphs.batch, args)
        elif self._encoder_type == 'Transformer': 
            nodes, node_padding_mask, edge_matrix, current_idx, action_idx, action_mask = state
            node_enhanced_padded = self.pre(q=nodes, key_padding_mask=node_padding_mask, attn_mask=edge_matrix) # Batch, Num_Key, Feature_Dim
        else:
            NotImplementedError
            
        current_idx = current_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim) # Batch, 1, Feature_Dim
        current_node = torch.gather(node_enhanced_padded, 1, current_idx) # Batch, 1, Feature_Dim

        action_idx = action_idx.unsqueeze(-1).repeat(1, 1, self._embedding_dim)
        action_node = torch.gather(node_enhanced_padded, 1, action_idx) # Batch, Num_Action, Feature_Dim
        
        # Q1
        # enhanced_current_node_1, attention_1 = self.decoder_1(current_node, action_node, action_mask) # enhance with action nodes
        enhanced_current_node_1, attention_1 = self.decoder_1(current_node, node_enhanced_padded, node_padding_mask) # enhance with all nodes
        enhanced_action_node_1 = torch.cat((enhanced_current_node_1.repeat(1, action_node.shape[1], 1), current_node.repeat(1, action_node.shape[1], 1), action_node), dim=-1)
        enhanced_action_node_1 = self.action_enhancing_1(enhanced_action_node_1)
        q_values_1 = self.q_value_embedding_1(enhanced_action_node_1)
        zero_1 = torch.zeros_like(q_values_1).cuda()
        q_values_masked_1 = torch.where(action_mask.unsqueeze(-1) == 0, zero_1, q_values_1) # Batch, Num_Action, 1

        return q_values_masked_1