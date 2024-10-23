#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.tools.buffer import ReplayBuffer_List, RolloutBuffer_vanilla, ReplayBuffer_Graph, RolloutBuffer

class RL_Policy(object):
    def __init__(self, args):
        
        if not hasattr(args, 'graph_task'):
            self.info_type = args.info_type
            self.action_type = args.action_type
            self.merge_vis = args.merge_vis
            self.max_action = args.max_action
            self.explore_noise = args.init_exploration_noise
            self.lr_scheduler_interval = args.lr_scheduler_interval
            self.random_exploration_length = args.random_exploration_length
            self.graph_task = args.graph_task
            self.surrouding_length = args.state_dim - args.vel_dim - args.goal_dim

            self.laser_dim = args.laser_dim
            self.laser_range = args.laser_range
            self.img_width = args.img_width
            self.img_height = args.img_height
            self.highlight_iterations = args.highlight_iterations
            
            self.state_dim = args.state_dim
            self.action_dim = args.action_dim
            self.discrete_action_dim = args.discrete_action_dim
            self.discrete_actions = args.discrete_actions
            self.discrete_action_v = args.discrete_action_v
            self.discrete_action_w = args.discrete_action_w
            self.observation_dim = args.observation_dim
            self.sample_length = args.sample_length
            
            self.buffer = ReplayBuffer_List(args.buffer_size, args.state_dim, args.action_dim, args.action_type)
        else:
            self.graph_using_pyg = args.graph_using_pyg
            self.random_exploration_length = args.random_exploration_length
            self.buffer = ReplayBuffer_Graph(args.buffer_size, args.graph_num_action_padding, args.graph_num_graph_padding, args.graph_node_feature_dim, args.graph_using_pyg)

    def update_buffer(self):
        return NotImplementedError

    def action_handler(self, action):
        return NotImplementedError
        
    def state_handler(self, state, if_batch=True):
        return NotImplementedError
    
    @torch.no_grad()
    def select_action(self, state):
        return NotImplementedError
