import math
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
"""


# Multihead Attention Block
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        '''
        input : batchsize, num_sets or num_seeds or num_inds -> (num), dim_feature
        output : batchsize, num_sets or num_seeds or num_inds -> (num), dim_V
        '''
        
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K) # dim_feature to dim_V

        # split dim_V to multi head, concatenate multi head in batch dimension
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0) # [batchsize * num_heads, num, dim_split]
        
        # weight matrix : [batchsize * num_heads, num_Q, num_K]
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_V), 2)
        # out : [batchsize, num_Q, dim_V]
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.leaky_relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


# Set Attention Block
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        '''
        INPUT  : [batch_size, num, dim]
        OUTPUT : [batch_size, num, dim]
        '''
        return self.mab(X, X)


# Induced # Set Attention Block
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        # low-rank projection : 
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        # reconstruction
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        '''
        INPUT  : [batch_size, num, dim]
        OUTPUT : [batch_size, num, dim_V]
        '''
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X) # [(batch_size, num_inds, dim_feature), (batch_size, num_sets, dim_feature)]
        return self.mab1(X, H)


# Pooling by Multihead Attention
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        '''
        INPUT  : [batch_size, num_seeds, dim_V]
        OUTPUT : [batch_size, num_seeds, dim]
        '''
        return self.mab(self.S.repeat(X.size(0), 1, 1), X) # [(batch_size, num_seeds, dim_feature), (batch_size, num_sets, dim_feature)]
    
    
class SetTransformer(nn.Module):
    def __init__(self, args, dim_input=4, num_seeds=1, dim_output=256,
            num_inds=8, dim_hidden=32, num_heads=2, ln=True):
        super(SetTransformer, self).__init__()
        
        self.num_agent = args.num_agent
        # original version
        # self.enc = nn.Sequential(
        #         ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
        #         ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, num_seeds, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Linear(dim_hidden, dim_output),
        #         nn.LeakyReLU())
        
        # simple version
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln))#,
                # SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.LeakyReLU())

    def forward(self, state):
        '''
        INPUT  : [batch_size, num_sets * dim_input]
        OUTPUT : [batch_size, num_seeds, dim_output] -> [batch_size, dim_output] num_seeds == 1
        '''
        state = state.reshape(-1, self.num_agent - 1, 4)
        
        # original version
        # out = self.dec(self.enc(state)).squeeze(1)
        
        # simple version
        out = self.enc(state)
        out = out.sum(dim=1, keepdim=True) # pooling mechanism: sum, mean, max
        out = self.dec(out).squeeze(1)
        
        return out
    
    