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

from navigation.habitat_action import HabitatAction

from env_tools.arguments import args as env_args


class SAC(RL_Policy):
    def __init__(self, args):
        super(SAC, self).__init__(args)

        self.actor = GraphPointerPolicy(node_dim=args.graph_node_feature_dim, 
                                        edge_dim=args.graph_edge_feature_dim,
                                        embedding_dim= args.graph_embedding_dim, 
                                        num_graph_padding=args.graph_num_graph_padding,
                                        encoder_type=args.graph_encoder).cuda()


        self.actor_old = GraphPointerPolicy(node_dim=args.graph_node_feature_dim, 
                                        edge_dim=args.graph_edge_feature_dim,
                                        embedding_dim= args.graph_embedding_dim, 
                                        num_graph_padding=args.graph_num_graph_padding,
                                        encoder_type=args.graph_encoder).cuda()

        self.critic = GraphQNet(node_dim=args.graph_node_feature_dim, 
                                edge_dim=args.graph_edge_feature_dim,
                                embedding_dim= args.graph_embedding_dim, 
                                num_graph_padding=args.graph_num_graph_padding,
                                encoder_type=args.graph_encoder).cuda()
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.graph_lr_actor)
        # self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr=args.graph_lr_critic)
        
        # self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        
        self.greedy = args.graph_sac_greedy
        self.lr_scheduler_interval = args.lr_scheduler_interval

        # entropy tuning
        self.lr_tune = args.lr_tune
        self.target_entropy = 0.05 * (-np.log(1 / args.graph_num_action_padding))

        self.discount = args.discount
        self.tau = args.tau
        
        self.train_step = 0 # 表示训练过程中经过的rl_step个数
        
        self.graph_num_action_padding = args.graph_num_action_padding
        self.args = args

        self.graph_iter_per_step = env_args.graph_iter_per_step
        self.eps_clip = 0.2

        self.MseLoss = nn.MSELoss()

        self.train_cnt = 0
        self.gamma = 0.99
    
    def update_buffer(self, state, action, reward, done, action_logprob, reward_arrive): # 向buffer中添加一个样本
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.reward_arrives.append(reward_arrive)


    
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
        new_state = self.state_handler(state, False)
        action_log_prob = self.actor_old(new_state, self.args).detach()  
        if self.greedy:
            action_index = torch.argmax(action_log_prob, dim=1).long()
        else:
            action_index = torch.multinomial(action_log_prob.exp(), 1).long().squeeze(1)

        action = state['action_idxes'][0, action_index.item()].cpu().numpy() # action在rl_topo中的index
        action_index = action_index.cpu().numpy() # action在action_space中的index
        action_log_prob_val = action_log_prob[0, action_index[0]]

        return action, action_index, action_log_prob_val # idx in padding


    def update_policy(self, writer):
        rewards = []
        discounted_reward = 0
        for reward, is_reward_arrive in zip(reversed(self.buffer.rewards), reversed(self.buffer.reward_arrives)):
            if is_reward_arrive:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # 归一化所有reward
        self.buffer.rewards = rewards

        # convert list to tensor
        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = self.buffer.logprobs

        for _ in range(self.graph_iter_per_step): # 将网络参数更新K次
            loss = self.train(writer)
        
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.train_cnt += 1

        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        writer.add_scalar('Training/Policy/loss', loss.mean().item(), self.train_cnt)



    def train(self, writer):
        self.actor.train()
        self.critic.train()

        '''load data batch'''
        state, action, reward, not_done, action_logprob = self.buffer.sample()
        state = self.state_handler(state)
        action = self.action_handler(action) # B,1 (选择到的action在action_space中的index)
        reward = reward.unsqueeze(1).float().cuda() # B,1,1
        not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
        action_logprob = action_logprob.unsqueeze(1).float().cuda() # B,1,1
        

        '''actor'''
        origin_logprobs = self.actor(state, self.args) # Batch, Num_Action # 用old_state得到对应的对数概率分布
        origin_logprobs = origin_logprobs.unsqueeze(-1).float().cuda() # B,N,1
        logprobs = torch.gather(origin_logprobs, 1, action.unsqueeze(-1)) # B,1,1
        
        origin_state_values = self.critic(state, self.args) # Batch, Num_Action, 1
        state_values = torch.gather(origin_state_values, 1, action.unsqueeze(-1)) # B,1,1(选择action对应的q值)
        
        entropy = (origin_logprobs * origin_logprobs.exp()).sum(dim=1) # (B,1)

        ratios = torch.exp(logprobs - action_logprob.detach()) # （B,1,1）

        advantages = reward - state_values.detach() # （B,1,1）
        surr1 = ratios * advantages # （B,1,1）
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages # （B,1,1）
        loss = -torch.min(surr1.squeeze(-1), surr2.squeeze(-1)) + 0.5*self.MseLoss(state_values.squeeze(-1), reward.squeeze(-1)) - 0.01*entropy
        
        # take gradient step
        self.actor_optimizer.zero_grad()
        loss.mean().backward()
        
        self.actor_optimizer.step()
        return loss
        

        # writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        # writer.add_scalar('Training/Policy/loss', loss.item(), self.train_cnt)

        # if self.train_step % self.lr_scheduler_interval == 0:
        #     self.scheduler_actor.step()
                    
    def save(self, dir_path):
        # save_models(self.critic, self.critic_optimizer, "critic", dir_path)
        save_models(self.actor, self.actor_optimizer, "actor", dir_path)

    def load(self, dir_path):
        # self.critic.load_state_dict(torch.load(dir_path + "_critic"))
        # self.critic_optimizer.load_state_dict(torch.load(dir_path + "_critic_optimizer"))
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(dir_path + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(dir_path + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actor)

    def load_il(self, dir_path):
        checkpoint = torch.load(dir_path, map_location="cuda")['model_state']
        self.actor.load_state_dict(checkpoint)
        self.actor_old.load_state_dict(checkpoint)



