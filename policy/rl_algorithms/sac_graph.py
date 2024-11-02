import random
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.distributions import MultivariateNormal

from policy.tools.utils import weights_init
from policy.tools.utils_network import FeatureMapper, soft_update, save_models

from policy.policy import RL_Policy
from policy.modules.graph_pointer import GraphPointerPolicy, GraphQNet
from torch_geometric.data.batch import Batch

from graph.tools import get_absolute_pos
# from policy.rl_algorithms.arguments import args


class SAC(RL_Policy):
    def __init__(self, args):
        super(SAC, self).__init__(args)

        self.actor = GraphPointerPolicy(node_dim=args.graph_node_feature_dim, 
                                        edge_dim=args.graph_edge_feature_dim,
                                        embedding_dim= args.graph_embedding_dim, 
                                        num_graph_padding=args.graph_num_graph_padding,
                                        encoder_type=args.graph_encoder).cuda()
        self.critic = GraphQNet(node_dim=args.graph_node_feature_dim, 
                                edge_dim=args.graph_edge_feature_dim,
                                embedding_dim= args.graph_embedding_dim, 
                                num_graph_padding=args.graph_num_graph_padding,
                                encoder_type=args.graph_encoder).cuda()
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.graph_lr_actor)
        self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr=args.graph_lr_critic)
        
        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)
        
        self.critic_loss = nn.MSELoss()
        
        self.greedy = args.graph_sac_greedy
        self.lr_scheduler_interval = args.lr_scheduler_interval

        # entropy tuning
        self.lr_tune = args.lr_tune
        self.alpha_init = args.alpha_init
        self.target_entropy = 0.05 * (-np.log(1 / args.graph_num_action_padding))
        
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=torch.float32, device=torch.device('cuda'))
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_tune)

        self.discount = args.discount
        self.tau = args.tau
        
        self.train_step = 0 # 表示训练过程中经过的rl_step个数
        
        self.graph_num_action_padding = args.graph_num_action_padding
        self.args = args
    
    def update_buffer(self, state, action, next_state, reward, done, p_idx):
        self.buffer.add(state, action, next_state, reward, done, p_idx)
    
    
    def state_handler(self, state, if_batch=True):    
        if self.graph_using_pyg:
            
            pyg_graph = state['pyg_graph'].cuda() if if_batch else Batch.from_data_list([state['pyg_graph']]).cuda()
            current_idx = state['current_idx'].cuda()
            action_idxes = state['action_idxes'].cuda()
            action_mask = state['action_mask'].cuda()
            
            return pyg_graph, current_idx, action_idxes, action_mask
        else:
            node_info_padded = state['node_info_padded'].cuda()
            node_padding_mask = state['node_padding_mask'].cuda()
            edge_matrix = state['edge_matrix'].cuda()
            current_idx = state['current_idx'].cuda()
            action_idxes = state['action_idxes'].cuda()
            action_mask = state['action_mask'].cuda()
            
            return node_info_padded, node_padding_mask, edge_matrix, current_idx, action_idxes, action_mask

    def action_handler(self, action):
        return action.cuda() # B num_action_node_padding
        
    @torch.no_grad()
    def select_action(self, state, if_train=False):
        if if_train and self.train_step < self.random_exploration_length: # random policy
            num_action = int(np.sum(state["action_mask"].cpu().numpy()))
            action_index = np.random.choice(num_action, 1)
            all_action_indexes = state['action_idxes'].squeeze(-1).cpu().numpy()[0]
            action = all_action_indexes[action_index][0]
        else:
            new_state = self.state_handler(state, False)
            action = self.actor(new_state, self.args).detach()  
            if self.greedy:
                action_index = torch.argmax(action, dim=1).long()
            else:
                action_index = torch.multinomial(action.exp(), 1).long().squeeze(1)
            action = state['action_idxes'][0, action_index.item()].cpu().numpy() # action在rl_topo中的index
            action_index = action_index.cpu().numpy() # action在action_space中的index
        return action, action_index # idx in padding


    def train(self, writer, train_index, batch_size=16):
        if(train_index==0):
            self.train_step += 1
        if self.train_step < self.random_exploration_length:
            return self.train_step

        self.actor.train()
        self.critic.train()

        '''load data batch'''
        state, action, next_state, reward, not_done, _ = self.buffer.sample(batch_size)
        state = self.state_handler(state)
        next_state = self.state_handler(next_state)
        
        action = self.action_handler(action)
        reward = reward.unsqueeze(1).float().cuda() # B,1,1
        not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
        
        '''critic'''
        with torch.no_grad():
            next_logprob = self.actor(next_state, self.args)
            target_q1, target_q2 = self.critic_target(next_state, self.args)
            next_q_values = torch.min(target_q1, target_q2)
            target_q = torch.sum(next_logprob.unsqueeze(2).exp() * (next_q_values - self.alpha * next_logprob.unsqueeze(2)), dim=1).unsqueeze(1)
            target_q = reward + self.discount * not_done * target_q
            
        all_q1, all_q2 = self.critic(state, self.args)
        current_q1 = torch.gather(all_q1, 1, action.unsqueeze(-1))
        current_q2 = torch.gather(all_q2, 1, action.unsqueeze(-1))
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()   
        
        '''actor'''
        logprob = self.actor(state, self.args)
        actor_loss = torch.sum((logprob.exp().unsqueeze(2) * (self.alpha * logprob.unsqueeze(2) - self.critic.Q1(state, self.args).detach())), dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()        

        '''automatic entropy tuning'''
        entropy = (logprob * logprob.exp()).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        soft_update(self.critic_target, self.critic, self.tau)

        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_step)
        writer.add_scalar('Training/Policy/actor_loss', actor_loss.item(), self.train_step)
        writer.add_scalar('Training/Policy/critic_loss', critic_loss.item(), self.train_step)
        writer.add_scalar('Training/Policy/alpha', self.alpha.detach().item(), self.train_step)
        
        if self.train_step % self.lr_scheduler_interval == 0:
            self.scheduler_actor.step()
            self.scheduler_critic.step()
        return self.train_step
                    
    def save(self, dir_path):
        save_models(self.critic, self.critic_optimizer, "critic", dir_path)
        save_models(self.actor, self.actor_optimizer, "actor", dir_path)

    def load(self, dir_path):
        self.critic.load_state_dict(torch.load(dir_path + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(dir_path + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(dir_path + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(dir_path + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def load_buffer_data(self, writer, load_buffer_data_cnt):
        while self.train_step<=load_buffer_data_cnt:
            load_dict = np.load(args.load_buffer_data_path+"{}.npy".format(self.train_step), allow_pickle=True).item()
            load_state = load_dict['state']
            load_action_indexes = load_dict['action_indexes']
            load_next_state = load_dict['next_state']
            load_reward = load_dict['reward']
            load_done = load_dict['done']

            policy.update_buffer(load_state, load_action_indexes, load_next_state, load_reward, load_done, 0) # 一个样本
            self.train_step += 1
            writer.add_scalar('Reward/reward_per_train_step', load_reward, self.train_step) # 记录每一个train_step的reward
