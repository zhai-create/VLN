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

    @torch.no_grad()
    def random_select_action(self, state, if_train=False):
        num_action = int(np.sum(state["action_mask"].cpu().numpy()))
        action_index = np.random.choice(num_action, 1)
        all_action_indexes = state['action_idxes'].squeeze(-1).cpu().numpy()[0]
        action = all_action_indexes[action_index][0]
        return action, action_index # idx in padding

    '''
    @torch.no_grad()
    def greedy_select_action(self, rl_graph, topo_graph):
        max_score = 0
        max_score_node = None
        candidate_intention_ls = []
        for temp_node in rl_graph.all_intention_nodes:
            if(temp_node.score>max_score):
                max_score = temp_node.score
                max_score_node = temp_node

            if(len(temp_node.near_score_ls)>0):
                if(temp_node.score<=np.mean(temp_node.near_score_ls)):
                    candidate_intention_ls.append(temp_node)
                
        if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
            min_dis = 10000
            res_node = None
            for temp_node in candidate_intention_ls:
                if(temp_node.parent_node.name==topo_graph.current_node.name):
                    temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                else:
                    temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                    temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])

                temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        else: # 没有“两次有效观察”的intention
            # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
            if(max_score>0.8):
                if(max_score_node.intention_type==1):
                    return max_score_node
            if(len(rl_graph.all_frontier_nodes)==0):
                return max_score_node
            min_dis = 10000
            res_node = None
            for temp_node in rl_graph.all_frontier_nodes:
                if(temp_node.parent_node.name==topo_graph.current_node.name):
                    temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                else:
                    temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                    temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])
                
                temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        return res_node
    '''

    '''
    @torch.no_grad()
    def greedy_select_action_ring_vlm_score(self, rl_graph, topo_graph):
        max_score = 0
        max_score_node = None
        candidate_intention_ls = []
        for temp_node in rl_graph.all_intention_nodes:
            if(temp_node.score>max_score):
                max_score = temp_node.score
                max_score_node = temp_node

            if(len(temp_node.near_score_ls)>0):
                if(temp_node.score<=np.mean(temp_node.near_score_ls)):
                    candidate_intention_ls.append(temp_node)
                
        if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
            min_dis = 10000
            res_node = None
            for temp_node in candidate_intention_ls:
                if(temp_node.parent_node.name==topo_graph.current_node.name):
                    temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                else:
                    temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                    temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])

                temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        else: # 没有“两次有效观察”的intention
            # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
            if(max_score>0.8):
                if(max_score_node.intention_type==1):
                    return max_score_node
            if(len(rl_graph.all_frontier_nodes)==0):
                return max_score_node

            # 选择一个frontier
            all_frontier_score = [temp_frontier.vlm_score for temp_frontier in rl_graph.all_frontier_nodes]
            all_zero_flag = np.all(np.array(all_frontier_score) == 0)
            if(all_zero_flag==True): # 选择距离机器人最近的frontier
                min_dis = 10000
                res_node = None
                for temp_node in rl_graph.all_frontier_nodes:
                    if(temp_node.parent_node.name==topo_graph.current_node.name):
                        temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                    else:
                        temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                        temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])
                    
                    temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                    if(temp_dis<min_dis):
                        min_dis = temp_dis
                        res_node = temp_node
            else: # 选择分数最高的frontier
                res_node = None
                max_score = 0

                now_frontier_distance_ls = []
                for temp_node in rl_graph.all_frontier_nodes:
                    if(temp_node.parent_node.name==topo_graph.current_node.name):
                        temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                    else:
                        temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                        temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])
                    temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                    now_frontier_distance_ls.append(temp_dis)
                
                sorted_nodes = [node for _, node in sorted(zip(now_frontier_distance_ls, rl_graph.all_frontier_nodes), key=lambda x: x[0])]
                for temp_node in sorted_nodes:
                    if(temp_node.vlm_score>max_score) or (res_node is None):
                        max_score = temp_node.vlm_score
                        res_node = temp_node

        return res_node
    '''


    @torch.no_grad()
    def greedy_select_action_ring_dis_score(self, rl_graph, topo_graph):
        max_score = 0
        max_score_node = None
        candidate_intention_ls = []
        for temp_node in rl_graph.all_intention_nodes:
            if(temp_node.score>max_score):
                max_score = temp_node.score
                max_score_node = temp_node

            if(len(temp_node.near_score_ls)>0):
                if(temp_node.score<=np.mean(temp_node.near_score_ls)):
                    candidate_intention_ls.append(temp_node)
                
        if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
            min_dis = 10000
            res_node = None
            for temp_node in candidate_intention_ls:
                if(temp_node.parent_node.name==topo_graph.current_node.name):
                    temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                else:
                    temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                    temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])

                temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        else: # 没有“两次有效观察”的intention
            # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
            if(max_score>0.8):
                if(max_score_node.intention_type==1):
                    return max_score_node
            if(len(rl_graph.all_frontier_nodes)==0):
                return max_score_node

            # 选择一个frontier
            all_frontier_score = [temp_frontier.vlm_score for temp_frontier in rl_graph.all_frontier_nodes]
            all_zero_flag = np.all(np.array(all_frontier_score) == 0)
            if(all_zero_flag==True): # 选择距离机器人最近的frontier
                min_dis = 10000
                res_node = None
                for temp_node in rl_graph.all_frontier_nodes:
                    if(temp_node.parent_node.name==topo_graph.current_node.name):
                        temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                    else:
                        temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                        temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])
                    
                    temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                    if(temp_dis<min_dis):
                        min_dis = temp_dis
                        res_node = temp_node
            else: # 选择分数最高的frontier
                res_node = None
                max_score = 0

                now_frontier_distance_ls = []
                for temp_node in rl_graph.all_frontier_nodes:
                    if(temp_node.parent_node.name==topo_graph.current_node.name):
                        temp_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                    else:
                        temp_parent_in_current_loc = topo_graph.current_node.all_other_nodes_loc[temp_node.parent_node.name]
                        temp_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_current_loc[:2], temp_parent_in_current_loc[2])
                    temp_dis = ((temp_loc[0]-topo_graph.rela_cx)**2+(temp_loc[1]-topo_graph.rela_cy)**2)**0.5
                    now_frontier_distance_ls.append(temp_dis)
                
                sorted_nodes = [node for _, node in sorted(zip(now_frontier_distance_ls, rl_graph.all_frontier_nodes), key=lambda x: x[0])]
                for temp_node in sorted_nodes:
                    if(temp_node.vlm_score>max_score) or (res_node is None):
                        max_score = temp_node.vlm_score
                        res_node = temp_node

        return res_node


    # 用于“our+rcnn+greedy+llm_frontier”
    '''
    @torch.no_grad()
    def greedy_select_action_frontier_score(self, rl_graph, world_cx, world_cy):
        max_score = 0
        max_score_node = None
        candidate_intention_ls = []
        for temp_node in rl_graph.all_intention_nodes:
            if(temp_node.score>max_score):
                max_score = temp_node.score
                max_score_node = temp_node

            if(len(temp_node.near_score_ls)>0):
                if(temp_node.score<=np.mean(temp_node.near_score_ls)):
                    candidate_intention_ls.append(temp_node)
                
        if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
            min_dis = 10000
            res_node = None
            for temp_node in candidate_intention_ls:
                temp_dis = ((temp_node.world_cx-world_cx)**2+(temp_node.world_cy-world_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        else: # 没有“两次有效观察”的intention
            # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
            if(max_score>0.8):
                if(max_score_node.intention_type==1):
                    return max_score_node
            if(len(rl_graph.all_frontier_nodes)==0):
                return max_score_node
            # 选择一个frontier
            all_frontier_score = [temp_frontier.vlm_score for temp_frontier in rl_graph.all_frontier_nodes]
            all_zero_flag = np.all(np.array(all_frontier_score) == 0)
            if(all_zero_flag==True): # 选择距离机器人最近的frontier
                min_dis = 10000
                res_node = None
                for temp_node in rl_graph.all_frontier_nodes:
                    temp_dis = ((temp_node.world_cx-world_cx)**2+(temp_node.world_cy-world_cy)**2)**0.5
                    if(temp_dis<min_dis):
                        min_dis = temp_dis
                        res_node = temp_node
            else: # 选择分数最高的frontier
                res_node = None
                max_score = 0
                now_frontier_distance_ls = [((temp_frontier.world_cx-world_cx)**2+(temp_frontier.world_cy-world_cy)**2)**0.5 for temp_frontier in rl_graph.all_frontier_nodes]
                # sorted_nodes = [node for _, node in sorted(zip(now_frontier_distance_ls, rl_graph.all_frontier_nodes))] # 按照距离递增排序后的list
                sorted_nodes = [node for _, node in sorted(zip(now_frontier_distance_ls, rl_graph.all_frontier_nodes), key=lambda x: x[0])]
                for temp_node in sorted_nodes:
                    if(temp_node.vlm_score>max_score) or (res_node is None):
                        max_score = temp_node.vlm_score
                        res_node = temp_node
        return res_node
    '''

    '''
    @torch.no_grad()
    def greedy_select_action_dis_score(self, rl_graph, world_cx, world_cy):
        max_score = 0
        max_score_node = None
        candidate_intention_ls = []
        for temp_node in rl_graph.all_intention_nodes:
            if(temp_node.score>max_score):
                max_score = temp_node.score
                max_score_node = temp_node

            if(len(temp_node.near_score_ls)>0):
                if(temp_node.score<=np.mean(temp_node.near_score_ls)):
                    candidate_intention_ls.append(temp_node)
                
        if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
            min_dis = 10000
            res_node = None
            for temp_node in candidate_intention_ls:
                temp_dis = ((temp_node.world_cx-world_cx)**2+(temp_node.world_cy-world_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    res_node = temp_node
        else: # 没有“两次有效观察”的intention
            # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
            if(max_score>0.8):
                if(max_score_node.intention_type==1):
                    return max_score_node
            if(len(rl_graph.all_frontier_nodes)==0):
                return max_score_node
            # 选择一个frontier
            now_frontier_distance_ls = [((temp_frontier.world_cx-world_cx)**2+(temp_frontier.world_cy-world_cy)**2)**0.5 for temp_frontier in rl_graph.all_frontier_nodes]
            sorted_nodes = [node for _, node in sorted(zip(now_frontier_distance_ls, rl_graph.all_frontier_nodes), key=lambda x: x[0])]
            res_node = None
            max_score = 0
            for temp_node in sorted_nodes:
                if(temp_node.vlm_score>max_score) or (res_node is None):
                    max_score = temp_node.vlm_score
                    res_node = temp_node
        return res_node
    '''



    # @torch.no_grad()
    # def greedy_select_action_near_goal(self, rl_graph, world_cx, world_cy, current_episode):
    #     # 找到距离当前机器人最近的object所在的位置
    #     min_goal_dis = 1000000
    #     min_goal_loc = None
    #     for temp_index in range(len(current_episode.goals)):
    #         temp_dis = ((current_episode.goals[temp_index].position[2]-world_cx)**2+(current_episode.goals[temp_index].position[0]-world_cy)**2)**0.5
    #         if temp_dis<min_goal_dis:
    #             min_goal_dis = temp_dis
    #             min_goal_loc = current_episode.goals[temp_index].position
        
    #     max_score = 0
    #     max_score_node = None
    #     candidate_intention_ls = []
    #     for temp_node in rl_graph.all_intention_nodes:
    #         if(temp_node.score>max_score):
    #             max_score = temp_node.score
    #             max_score_node = temp_node

    #         if(len(temp_node.near_score_ls)>0):
    #             if(temp_node.score<=np.mean(temp_node.near_score_ls)):
    #                 candidate_intention_ls.append(temp_node)
                
    #     if(len(candidate_intention_ls)>0): # 具有“两次有效观察”的intention
    #         min_dis = 10000
    #         res_node = None
    #         for temp_node in candidate_intention_ls:
    #             temp_dis = ((temp_node.world_cx-world_cx)**2+(temp_node.world_cy-world_cy)**2)**0.5
    #             if(temp_dis<min_dis):
    #                 min_dis = temp_dis
    #                 res_node = temp_node
    #     else: # 没有“两次有效观察”的intention
    #         # if(max_score>0.8) or (len(rl_graph.all_frontier_nodes)==0):
    #         if(max_score>0.8):
    #             if(max_score_node.intention_type==1):
    #                 return max_score_node
    #         if(len(rl_graph.all_frontier_nodes)==0):
    #             return max_score_node
    #         min_dis = 10000
    #         res_node = None
    #         for temp_node in rl_graph.all_frontier_nodes:
    #             temp_dis = ((temp_node.world_cx-min_goal_loc[0])**2+(temp_node.world_cy-min_goal_loc[1])**2)**0.5
    #             if(temp_dis<min_dis):
    #                 min_dis = temp_dis
    #                 res_node = temp_node
    #     return res_node
                


    def train(self, writer, train_index, batch_size=16):
        if(train_index==0):
            self.train_step += 1
            HabitatAction.episode_train_step += 1
        if self.train_step < self.random_exploration_length:
            return self.train_step

        self.actor.train()
        self.critic.train()

        '''load data batch'''
        state, action, next_state, reward, not_done, _ = self.buffer.sample(batch_size)
        state = self.state_handler(state)
        next_state = self.state_handler(next_state)
        
        action = self.action_handler(action) # B,1 (选择到的action在action_space中的index)
        reward = reward.unsqueeze(1).float().cuda() # B,1,1
        not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
        
        '''critic'''
        with torch.no_grad():
            next_logprob = self.actor(next_state, self.args) # Batch, Num_Action
            target_q1, target_q2 = self.critic_target(next_state, self.args) # Batch, Num_Action, 1
            next_q_values = torch.min(target_q1, target_q2) # Batch, Num_Action, 1
            target_q = torch.sum(next_logprob.unsqueeze(2).exp() * (next_q_values - self.alpha * next_logprob.unsqueeze(2)), dim=1).unsqueeze(1)
            target_q = reward + self.discount * not_done * target_q # B, 1, 1
            
        all_q1, all_q2 = self.critic(state, self.args) # Batch, Num_Action, 1
        current_q1 = torch.gather(all_q1, 1, action.unsqueeze(-1)) # B,1,1(选择action对应的q值)
        current_q2 = torch.gather(all_q2, 1, action.unsqueeze(-1)) # B,1,1
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()   
        
        '''actor'''
        logprob = self.actor(state, self.args) # Batch, Num_Action
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

    def load_il(self, dir_path):
        checkpoint = torch.load(dir_path, map_location="cuda")['model_state']
        self.actor.load_state_dict(checkpoint)


    def load_buffer_data(self, writer, load_buffer_data_cnt, load_buffer_data_path):
        while self.train_step<=load_buffer_data_cnt:
            load_dict = np.load("{}/{}.npy".format(load_buffer_data_path, self.train_step), allow_pickle=True).item()
            load_state = load_dict['current_state']
            load_action_indexes = load_dict['policy_acton_idx']
            load_next_state = load_dict['next_state']
            load_reward = load_dict['reward']
            load_done = load_dict['done']

            self.update_buffer(load_state, load_action_indexes, load_next_state, load_reward, load_done, 0) # 一个样本
            self.train_step += 1
            writer.add_scalar('Result/reward_per_rl_step', load_reward, self.train_step) # 记录每一个train_step的reward
