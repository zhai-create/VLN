import os
import numpy as np
from copy import deepcopy
from policy.tools.utils_graph import padding_list, to_adjacency_matrix

import torch
import torch_geometric.nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

####################################
#          Replay Buffer
####################################

class ReplayBuffer_Backup(object):
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.save_time = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.cbf_label = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done, cbf_label, num):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.cbf_label[self.ptr] = cbf_label

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.size % 500 == 0 and self.size != self.max_size:
            print("=========================>  replay buffer size :  {}".format(self.size))
            
    def save_file(self, dir, state, action, next_state, reward, done, cbf_label):
        if not done:
            # state
            temp_state = state[:]
            temp_state_array = np.asarray(temp_state)
            state_dir = dir + 'state/'
            os.makedirs(state_dir, exist_ok=True)
            np.save(state_dir + str(self.save_time).zfill(7), temp_state_array)
            # action
            temp_action = action[:]
            temp_action_array = np.asarray(temp_action)
            action_dir = dir + 'action/'
            os.makedirs(action_dir, exist_ok=True)
            np.save(action_dir + str(self.save_time).zfill(7), temp_action_array)
            # next state
            temp_next_state = next_state[:]
            temp_next_state_array = np.asarray(temp_next_state)
            nstate_dir = dir + 'next_state/'
            os.makedirs(nstate_dir, exist_ok=True)
            np.save(nstate_dir + str(self.save_time).zfill(7), temp_next_state_array)
            # reward
            temp_reward = reward
            temp_reward_array = np.asarray(temp_reward)
            reward_dir = dir + 'reward/'
            os.makedirs(reward_dir, exist_ok=True)
            np.save(reward_dir + str(self.save_time).zfill(7), temp_reward_array)
            
            # cbf_label
            temp_cbf_label = cbf_label
            temp_cbf_label_array = np.asarray(temp_cbf_label)
            cbf_label_dir = dir + 'cbf_label/'
            os.makedirs(cbf_label_dir, exist_ok=True)
            np.save(cbf_label_dir + str(self.save_time).zfill(7), temp_cbf_label_array)

        self.save_time += 1

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.state[ind][:],
                self.action[ind][:],
                self.next_state[ind][:],
                self.reward[ind][:],
                self.not_done[ind][:],
                self.cbf_label[ind][:])

class ReplayBuffer_List(object):
    def __init__(self, max_size, s_dim, a_dim, action_type):        
        self._storage = {}
        self._max_size = int(max_size)
        self._current_size = 0
        self._next_idx = 0
        self._save_time = 0
        self._s_dim = s_dim
        self._a_dim = a_dim
        self._action_type = action_type
        
        
    def __len__(self):
        return min(self._current_size, self._max_size)

    def add(self, state, action, next_state, reward, done, cbf_label, num_set=0):
        
        try:
            len(self._storage[num_set])
        except:
            self._storage[num_set] = []
        
        data = (deepcopy(state), deepcopy(action), deepcopy(next_state), reward, 1. - done, cbf_label)
        
        if self._next_idx >= len(self._storage[num_set]):
            self._storage[num_set].append(data)
        else:
            self._storage[num_set][self._next_idx] = data
            
        self._next_idx = int((self._current_size + 1) % self._max_size)
        self._current_size += 1

    def sample(self, batch_size, num_set=0):
        idxes = np.random.randint(0, len(self._storage[num_set])-1, size=batch_size)
        
        batch_state = np.zeros((batch_size, self._s_dim))
        batch_action = np.zeros((batch_size, self._a_dim)) if self._action_type == 'continuous' else np.zeros((batch_size, 1))
        batch_next_state = np.zeros((batch_size, self._s_dim))
        batch_reward = np.zeros((batch_size, 1))
        batch_not_done = np.zeros((batch_size, 1))
        batch_cbf_label = np.zeros((batch_size, 1))
        
        # np.array(list(itemgetter(*idxes)(self.state)))[:]
        for i, idx in enumerate(idxes):
            data = self._storage[num_set][idx]
            state, action, next_state, reward, not_done, cbf_label = data

            batch_state[i] = deepcopy(state)
            batch_action[i] = deepcopy(action)
            batch_next_state[i] = deepcopy(next_state)
            batch_reward[i] = reward
            batch_not_done[i] = not_done
            batch_cbf_label[i] = cbf_label
        return (batch_state, batch_action, batch_next_state, batch_reward, batch_not_done, batch_cbf_label)


class ReplayBuffer_Graph(object):
    def __init__(self, max_size, num_action_padding=5, num_graph_padding=5, node_feature_dim=3, using_pyg=True):        
        self._storage = {}
        self._max_size = int(max_size)
        self._current_size = 0
        self._next_idx = 0
        self._save_time = 0       
        self._num_action_padding = num_action_padding 
        self._num_graph_padding= num_graph_padding
        self._using_pyg = using_pyg
        
        self._node_feature_dim = node_feature_dim
        
    def __len__(self):
        return min(self._current_size, self._max_size)

    def add(self, state, action, next_state, reward, done, num_set=0):
        
        try:
            len(self._storage[num_set])
        except:
            self._storage[num_set] = []

        if self._using_pyg:               
            data = (dict(pyg_graph=state['pyg_graph'], current_idx=state['current_idx'], action_idxes=state['action_idxes'], action_mask=state['action_mask']),
                    torch.tensor(action), 
                    dict(pyg_graph=next_state['pyg_graph'], current_idx=next_state['current_idx'], action_idxes=next_state['action_idxes'], action_mask=next_state['action_mask']),
                    reward, 
                    1. - done)
        else:
            data = (dict(node_info_padded=state['node_info_padded'], node_padding_mask=state['node_padding_mask'], edge_matrix=state['edge_matrix'], 
                         current_idx=state['current_idx'], action_idxes=state['action_idxes'], action_mask=state['action_mask']),
                    torch.tensor(action), 
                    dict(node_info_padded=next_state['node_info_padded'], node_padding_mask=next_state['node_padding_mask'], edge_matrix=next_state['edge_matrix'], 
                         current_idx=next_state['current_idx'], action_idxes=next_state['action_idxes'], action_mask=next_state['action_mask']),
                    reward, 
                    1. - done)
        
        if self._next_idx >= len(self._storage[num_set]):
            self._storage[num_set].append(data)
        else:
            self._storage[num_set][self._next_idx] = data
            
        self._next_idx = int((self._current_size + 1) % self._max_size)
        self._current_size += 1

    def sample(self, batch_size, num_set=0):
        # print('self._storage[num_set]:', self._storage[num_set])
        # print('len(self._storage[num_set]):', len(self._storage[num_set]))
        # idxes = np.random.randint(0, len(self._storage[num_set])-1, size=batch_size)
        idxes = np.random.randint(0, len(self._storage[num_set]), size=batch_size)
        if self._using_pyg:
            batch_graph = []
            batch_current_idx = torch.zeros((batch_size, 1))
            batch_action_idxes = torch.zeros((batch_size, self._num_action_padding))
            batch_action_mask = torch.zeros((batch_size, self._num_action_padding))

            batch_action = torch.zeros((batch_size, 1))
            
            batch_next_graph = []
            batch_next_current_idx = torch.zeros((batch_size, 1))
            batch_next_action_idxes = torch.zeros((batch_size, self._num_action_padding))
            batch_next_action_mask = torch.zeros((batch_size, self._num_action_padding))
            
            batch_reward = torch.zeros((batch_size, 1))
            batch_not_done = torch.zeros((batch_size, 1))
            for i, idx in enumerate(idxes):
                data = self._storage[num_set][idx]
                
                
                state, action, next_state, reward, not_done = data
                                
                batch_graph.append(state['pyg_graph'])
                batch_current_idx[i] = state['current_idx']
                batch_action_idxes[i] = state['action_idxes']
                batch_action_mask[i] = state['action_mask']
                
                batch_action[i] = deepcopy(action)
                
                batch_next_graph.append(next_state['pyg_graph'])
                batch_next_current_idx[i] = next_state['current_idx']
                batch_next_action_idxes[i] = next_state['action_idxes']
                batch_next_action_mask[i] = next_state['action_mask']
                
                batch_reward[i] = reward
                batch_not_done[i] = not_done
            # print("==================before return==================")
            # print(Batch.from_data_list(batch_graph))
            # print("==================after return==================")
            return (dict(pyg_graph=Batch.from_data_list(batch_graph), current_idx=batch_current_idx.long(), action_idxes=batch_action_idxes.long(), action_mask=batch_action_mask),
                    batch_action.long(), 
                    dict(pyg_graph=Batch.from_data_list(batch_next_graph), current_idx=batch_next_current_idx.long(), action_idxes=batch_next_action_idxes.long(), action_mask=batch_next_action_mask),
                    batch_reward, batch_not_done, None)
        else:                                                                                                        
            batch_node_info_padded = torch.zeros((batch_size, self._num_graph_padding, self._node_feature_dim))
            batch_node_padding_mask = torch.zeros((batch_size, self._num_graph_padding))
            batch_edge_matrix = torch.zeros((batch_size, self._num_graph_padding, self._num_graph_padding))
            batch_current_idx = torch.zeros((batch_size, 1))
            batch_action_idxes = torch.zeros((batch_size, self._num_action_padding))
            batch_action_mask = torch.zeros((batch_size, self._num_action_padding))
            
            batch_action = torch.zeros((batch_size, 1))
            
            batch_next_node_info_padded = torch.zeros((batch_size, self._num_graph_padding, self._node_feature_dim))
            batch_next_node_padding_mask = torch.zeros((batch_size, self._num_graph_padding))
            batch_next_edge_matrix = torch.zeros((batch_size, self._num_graph_padding, self._num_graph_padding))
            batch_next_current_idx = torch.zeros((batch_size, 1))
            batch_next_action_idxes = torch.zeros((batch_size, self._num_action_padding))
            batch_next_action_mask = torch.zeros((batch_size, self._num_action_padding))
            
            batch_reward = torch.zeros((batch_size, 1))
            batch_not_done = torch.zeros((batch_size, 1))
            for i, idx in enumerate(idxes):
                data = self._storage[num_set][idx]
                
                state, action, next_state, reward, not_done = data
                batch_node_info_padded[i] = state['node_info_padded'][0]
                batch_node_padding_mask[i] = state['node_padding_mask'][0]
                batch_edge_matrix[i] = state['edge_matrix'][0]
                batch_current_idx[i] = state['current_idx']
                batch_action_idxes[i] = state['action_idxes']
                batch_action_mask[i] = state['action_mask']
                
                batch_action[i] = deepcopy(action)
                
                batch_next_node_info_padded[i] = next_state['node_info_padded'][0]
                batch_next_node_padding_mask[i] = next_state['node_padding_mask'][0]
                batch_next_edge_matrix[i] = next_state['edge_matrix'][0]
                batch_next_current_idx[i] = next_state['current_idx']
                batch_next_action_idxes[i] = next_state['action_idxes']
                batch_next_action_mask[i] = next_state['action_mask']
                
                batch_reward[i] = reward
                batch_not_done[i] = not_done

            return (dict(node_info_padded=batch_node_info_padded, node_padding_mask=batch_node_padding_mask, edge_matrix=batch_edge_matrix, 
                         current_idx=batch_current_idx.long(), action_idxes=batch_action_idxes.long(), action_mask=batch_action_mask),
                    batch_action.long(), 
                    dict(node_info_padded=batch_next_node_info_padded, node_padding_mask=batch_next_node_padding_mask, edge_matrix=batch_next_edge_matrix, 
                         current_idx=batch_next_current_idx.long(), action_idxes=batch_next_action_idxes.long(), action_mask=batch_next_action_mask),
                    batch_reward, batch_not_done, None)


class RolloutBuffer_vanilla(object):
    def __init__(self, batch_size, train_episode_length, s_dim, a_dim, action_type, gae, gamma): 
        self._storage = {} # (data_idx, step, 7)
        self._train_episode_length = int(train_episode_length)
        self._batch_size = batch_size
        self._current_size = 0
        self._save_time = 0
        self._s_dim = s_dim
        self._a_dim = a_dim
        self._action_type = action_type
        self._gae = gae 
        self._gamma = gamma
        self._gae_lamda = 0.95
        self._sample_step = 0
                 

    def __len__(self):
        return len(self._storage[0])
    
    def add(self, state, action, logprob, value, reward, done, cbf_label, data_idx=0):
        
        try:
            len(self._storage[data_idx])
        except:
            self._storage[data_idx] = []
        
        data = (state.copy(), action.copy(), logprob, reward, 1. - done, value, cbf_label)
        
        self._storage[data_idx].append(data)            
        self._current_size += 1
        
        
    def reward2return_vanilla(self, data_idx=0):
        # Only for buffer with only single agent's data!!!
        rewards = []
        discounted_reward = 0
        for _, _, _, _, reward, not_done, _ in reversed(self._storage[data_idx]):
            if not not_done:
                discounted_reward = 0
            discounted_reward = reward + self._gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        for idx in range(len(rewards)):
            state, action, logprob, value, reward, not_done, cbf_label = self._storage[data_idx][idx]
            self._storage[data_idx][idx] = (state, action, logprob, value, rewards[idx], not_done, cbf_label)    
        return
    
        
    def clear(self, num_set=0):
        del self._storage[num_set]
        self._current_size = 0
        self._storage[num_set] = []
        self.rollout_reward = False
        self._sample_step = 0
        

    def sample_vanilla(self, num_set=0):
        if_random = True
        batch_size = len(self) if self._batch_size <= 0 else self._batch_size
        
        if if_random: 
            idxes = np.random.randint(0, len(self._storage[num_set])-1, size=batch_size)
            batch_state = np.zeros((batch_size, self._s_dim))
            batch_action = np.zeros((batch_size, self._a_dim)) if self._action_type == 'continuous' else np.zeros((batch_size, 1))
            batch_reward = np.zeros((batch_size, 1))
            
            for i, idx in enumerate(idxes):
                data = self._storage[num_set][idx]
                state, action, _, _, reward, _, _ = data

                batch_state[i] = state
                batch_action[i] = action
                batch_reward[i] = reward
                
            
        self._sample_step += 1
        return (batch_state, batch_action, batch_reward)
    
    
class RolloutBuffer(object):
    def __init__(self, batch_size, train_episode_length, s_dim, a_dim, action_type, gae, gamma, gae_lamda):         
        self._state_buffer = {}
        self._action_buffer = {}
        self._logprob_buffer = {}
        self._reward_buffer = {}
        self._not_done_buffer = {}
        self._value_buffer = {}
        
        self._advantage_buffer = {}
        self._return_buffer = {}
        
        self._train_episode_length = int(train_episode_length)
        self._action_type = action_type
        self._batch_size = batch_size
        self._s_dim = s_dim
        self._a_dim = a_dim
        self._gamma = gamma
        self._gae = gae 
        
        self._gae_lamda = gae_lamda
        self._sample_step = 0
        self._save_time = 0
                 

    def __len__(self):
        length = []
        for i in range(len(self._state_buffer)):
            length.append(len(self._state_buffer[i]))
        return max(length)
    
    
    def clear(self):
        for i in range(len(self._state_buffer)):
            del self._state_buffer[i]
            del self._action_buffer[i]
            del self._logprob_buffer[i]
            del self._reward_buffer[i]
            del self._not_done_buffer[i]
            del self._value_buffer[i]
            del self._advantage_buffer[i]
            del self._return_buffer[i]
            
            self._state_buffer[i] = []
            self._action_buffer[i] = []
            self._logprob_buffer[i] = []
            self._reward_buffer[i] = []
            self._not_done_buffer[i] = []
            self._value_buffer[i] = []
            self._advantage_buffer[i] = []
            self._return_buffer[i] = []
        
    
    def add(self, state, action, logprob, value, reward, done, cbf_label, data_idx=0):
        try:
            len(self._state_buffer[data_idx])
            len(self._action_buffer[data_idx])
            len(self._logprob_buffer[data_idx])
            len(self._reward_buffer[data_idx])
            len(self._not_done_buffer[data_idx])
            len(self._value_buffer[data_idx])
        except:
            self._state_buffer[data_idx] = []
            self._action_buffer[data_idx] = []
            self._logprob_buffer[data_idx] = []
            self._reward_buffer[data_idx] = []
            self._not_done_buffer[data_idx] = []
            self._value_buffer[data_idx] = []
            
        self._state_buffer[data_idx].append(state.copy())
        self._action_buffer[data_idx].append(action.copy())
        self._logprob_buffer[data_idx].append(logprob)
        self._reward_buffer[data_idx].append(reward)
        self._not_done_buffer[data_idx].append(1. - done)
        self._value_buffer[data_idx].append(value)
            
    
    # Bootstrap with Generalized Advantage Estimation (GAE)
    def reward2return(self):
        for data_idx in range(len(self._state_buffer)):
            # use the last data as init
            next_not_done = self._not_done_buffer[data_idx][-1]
            next_value = self._value_buffer[data_idx][-1]
            if self._gae:
                last_gae_lamda = 0
                advantages = np.zeros(len(self._state_buffer[data_idx])-1)
                for t in reversed(range(len(self._state_buffer[data_idx])-1)):
                    if t == len(self._state_buffer[data_idx]) - 2:
                        nextnonterminal = next_not_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = self._not_done_buffer[data_idx][t + 1]
                        nextvalues = self._value_buffer[data_idx][t + 1]
                        
                    delta = self._reward_buffer[data_idx][t] + self._gamma * nextvalues * nextnonterminal - self._value_buffer[data_idx][t]  
                    advantages[t] = last_gae_lamda = delta + self._gamma * self._gae_lamda * nextnonterminal * last_gae_lamda
                returns = advantages + self._value_buffer[data_idx][0:(len(self._value_buffer[data_idx])-1)]
            else:
                returns = np.zeros(len(self._state_buffer[data_idx])-1)
                for t in reversed(range(len(self._state_buffer[data_idx])-1)):
                    if t == len(self._state_buffer[data_idx]) - 2:
                        nextnonterminal = next_not_done
                        next_return = next_value
                    else:
                        nextnonterminal = self._not_done_buffer[data_idx][t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self._reward_buffer[data_idx][t] + self._gamma * nextnonterminal * next_return
                advantages = returns - self._value_buffer[data_idx][0:(len(self._value_buffer[data_idx])-1)]

            self._advantage_buffer[data_idx] = advantages
            self._return_buffer[data_idx] = returns
            
    
    def sample_all_batch(self):
        # Bootstrap
        self.reward2return()
        
        # Combine all data  
        list_states = []
        list_actions = []
        list_logprobs = []
        list_advantages = []
        list_returns = []
        list_values = []
        
        for data_idx in range(len(self._state_buffer)):
            t_states = self._state_buffer[data_idx][0:(len(self._state_buffer[data_idx])-1)]
            t_actions = self._action_buffer[data_idx][0:(len(self._action_buffer[data_idx])-1)]
            t_logprobs = self._logprob_buffer[data_idx][0:(len(self._logprob_buffer[data_idx])-1)]
            t_advantages = self._advantage_buffer[data_idx][0:(len(self._advantage_buffer[data_idx]))]
            t_returns = self._return_buffer[data_idx][0:(len(self._return_buffer[data_idx]))]
            t_values = self._value_buffer[data_idx][0:(len(self._value_buffer[data_idx])-1)]
            
            list_states.append(np.array(t_states.copy()))
            list_actions.append(np.array(t_actions.copy()))
            list_logprobs.append(np.array(t_logprobs.copy()))
            list_advantages.append(np.array(t_advantages.copy()))
            list_returns.append(np.array(t_returns.copy()))
            list_values.append(np.array(t_values.copy()))
        
        b_states = np.concatenate(list_states, axis=0).reshape(-1, self._s_dim)
        b_actions = np.concatenate(list_actions, axis=0).reshape(-1, self._a_dim)
        b_logprobs = np.concatenate(list_logprobs, axis=0).reshape(-1, 1)
        b_advantages = np.concatenate(list_advantages, axis=0).reshape(-1, 1)
        b_returns = np.concatenate(list_returns, axis=0).reshape(-1, 1)
        b_values = np.concatenate(list_values, axis=0).reshape(-1, 1)
        
        return b_states, b_actions, b_logprobs, b_advantages, b_returns, b_values 
