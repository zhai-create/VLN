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

        # self.critic_old = GraphQNet(node_dim=args.graph_node_feature_dim, 
        #                         edge_dim=args.graph_edge_feature_dim,
        #                         embedding_dim= args.graph_embedding_dim, 
        #                         num_graph_padding=args.graph_num_graph_padding,
        #                         encoder_type=args.graph_encoder).cuda()
    
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.graph_lr_actor, eps=1e-5)
        self.critic_optimizer= torch.optim.Adam(self.critic.parameters(), lr=args.graph_lr_critic, eps=1e-5)
        
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
        self.lamda = 0.95

        self.select_action_cnt = 0

        self.vf_coef = 0.5
        self.ent_coef = 0.01

        self.minibatch_size = args.minibatch_size
    
    def update_buffer(self, state, action, next_state, reward, done, action_logprob, reward_arrive): # 向buffer中添加一个样本
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.reward_arrives.append(reward_arrive)
        self.buffer.next_states.append(next_state)

    def delete_buffer(self):
        while (self.buffer.reward_arrives) and (self.buffer.reward_arrives[-1] is False):
            # 删除最后一个元素
            self.buffer.states.pop()
            self.buffer.actions.pop()
            self.buffer.rewards.pop()
            self.buffer.is_terminals.pop()
            self.buffer.logprobs.pop()
            self.buffer.reward_arrives.pop()
            self.buffer.next_states.pop()
    
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
    def select_action(self, writer, state, if_train=False):
        new_state = self.state_handler(state, False)
        action_log_prob = self.actor_old(new_state, self.args).detach()  

        action_std = action_log_prob.exp().std().item()
        writer.add_scalar('Training/Policy/action_std', action_std, self.select_action_cnt)
        self.select_action_cnt += 1

        if self.greedy:
            action_index = torch.argmax(action_log_prob, dim=1).long()
        else:
            action_index = torch.multinomial(action_log_prob.exp(), 1).long().squeeze(1)

        action = state['action_idxes'][0, action_index.item()].cpu().numpy() # action在rl_topo中的index
        action_index = action_index.cpu().numpy() # action在action_space中的index
        action_log_prob_val = action_log_prob[0, action_index[0]]

        return action, action_index, action_log_prob_val # idx in padding

    '''
    def update_policy(self, writer):
        rewards = []
        discounted_reward = 0
        temp_index = 0
        for reward, is_reward_arrive in zip(reversed(self.buffer.rewards), reversed(self.buffer.reward_arrives)):
            if (is_reward_arrive==True):
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

            if(temp_index==0):
                assert (is_reward_arrive==True)
            temp_index += 1

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to("cuda")
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # 归一化所有reward
        self.buffer.rewards = rewards

        # # convert list to tensor
        # old_states = self.buffer.states
        # old_actions = self.buffer.actions
        # old_logprobs = self.buffer.logprobs

        all_old_states, all_old_actions, all_old_reward, all_old_not_done, all_old_logprobs = self.buffer.sample()
        # all_old_states = self.state_handler(all_old_states)


        all_old_pyg_graph = Batch.from_data_list(all_old_states['pyg_graph']).cuda()
        all_old_current_idx = all_old_states['current_idx'].cuda()
        all_old_action_idxes = all_old_states['action_idxes'].cuda()
        all_old_action_mask = all_old_states['action_mask'].cuda()

        all_old_actions = self.action_handler(all_old_actions) # B,1 (选择到的action在action_space中的index)
        all_old_logprobs = all_old_logprobs.unsqueeze(1).float().cuda() # B,1,1

        
        with torch.no_grad():
            # q_values_from_old_critic: (batch_size, Num_Action, 1)
            v_values_from_old_critic = self.critic_old((all_old_pyg_graph, all_old_current_idx, all_old_action_idxes, all_old_action_mask), self.args) # 使用旧critic
            # baseline_v_values: (batch_size, 1) - 已采取动作对应的旧Q值
            baseline_v_values = v_values_from_old_critic.squeeze(2)

        advantages = rewards - baseline_v_values.detach() # （B,1,1）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

        # 步骤 3: PPO Epoch 循环和 Minibatch 迭代
        # --------------------------------------------------------------------
        num_samples_in_rollout = len(self.buffer.states) # 应该是 rl_args.update_timestep (1024)
        accumulated_loss = 0.0
        num_minibatch_updates = 0

        for _ in range(self.graph_iter_per_step): # PPO Epochs
            permutation_indices = np.random.permutation(num_samples_in_rollout)

            for start_idx in range(0, num_samples_in_rollout, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                mb_indices = permutation_indices[start_idx:end_idx]

                # 提取当前minibatch的数据
                # mb_states = all_old_states[mb_indices]

                # # 使用列表推导式对元组中的每个张量应用高级索引
                # minibatch_states_list = [state_part[mb_indices] for state_part in all_old_states]
                # # 将列表转换回元组
                # mb_states = tuple(minibatch_states_list)

                part_old_pyg_graph = Batch.from_data_list([all_old_states['pyg_graph'][temp_index] for temp_index in mb_indices]).cuda()
                part_old_current_idx = all_old_current_idx[mb_indices]
                part_old_action_idxes = all_old_action_idxes[mb_indices]
                part_old_action_mask = all_old_action_mask[mb_indices]
                mb_states = (part_old_pyg_graph, part_old_current_idx, part_old_action_idxes, part_old_action_mask)

                mb_actions = all_old_actions[mb_indices]
                mb_old_logprobs = all_old_logprobs[mb_indices]
                mb_returns_G_t = rewards[mb_indices] # 当前minibatch的G_t
                mb_advantages = advantages[mb_indices]
        
                # 3b. 调用 train 函数处理这个minibatch
                loss_this_minibatch = self.train(
                    writer,
                    mb_states,
                    mb_actions,
                    mb_old_logprobs,
                    mb_returns_G_t, # 将G_t作为"reward"参数传递给train函数
                    mb_advantages
                )
                accumulated_loss += loss_this_minibatch.mean().item()
                num_minibatch_updates += 1
        
        avg_loss_for_update = accumulated_loss / num_minibatch_updates if num_minibatch_updates > 0 else 0
        

        # for _ in range(self.graph_iter_per_step): # 将网络参数更新K次
        #     loss = self.train(writer)
        
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        self.train_cnt += 1
        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        writer.add_scalar('Training/Policy/avg_loss_per_update', avg_loss_for_update, self.train_cnt)
    '''

    def update_policy(self, writer):
        temp_index = 0
        for reward, is_reward_arrive in zip(reversed(self.buffer.rewards), reversed(self.buffer.reward_arrives)):
            if(temp_index==0):
                assert (is_reward_arrive==True)
                break
            temp_index += 1
        
        all_old_states, all_old_actions, all_old_next_state, all_old_reward, all_old_not_done, all_old_logprobs = self.buffer.sample()

        # =====> state <=====
        all_old_pyg_graph = Batch.from_data_list(all_old_states['pyg_graph']).cuda()
        all_old_current_idx = all_old_states['current_idx'].cuda()
        all_old_action_idxes = all_old_states['action_idxes'].cuda()
        all_old_action_mask = all_old_states['action_mask'].cuda()

        # =====> next_state <=====
        all_old_next_pyg_graph = Batch.from_data_list(all_old_next_state['pyg_graph']).cuda()
        all_old_next_current_idx = all_old_next_state['current_idx'].cuda()
        all_old_next_action_idxes = all_old_next_state['action_idxes'].cuda()
        all_old_next_action_mask = all_old_next_state['action_mask'].cuda()

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic((all_old_next_pyg_graph, all_old_next_current_idx, all_old_next_action_idxes, all_old_next_action_mask), self.args) # 使用旧critic
            vs = vs.squeeze(2).cuda()

            vs_ = self.critic((all_old_next_pyg_graph, all_old_next_current_idx, all_old_next_action_idxes, all_old_next_action_mask), self.args)
            vs_ = vs_.squeeze(2).cuda()

            # deltas = all_old_reward + self.gamma * all_old_not_done * vs_ - vs
            deltas = all_old_reward.cuda() + self.gamma * all_old_not_done.cuda() * vs_ - vs
            for delta, not_d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(all_old_not_done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * not_d
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).cuda()
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # 步骤 3: PPO Epoch 循环和 Minibatch 迭代
        # --------------------------------------------------------------------
        num_samples_in_rollout = len(self.buffer.states) # 应该是 rl_args.update_timestep (1024)
        accumulated_actor_loss = 0.0
        accumulated_critic_loss = 0.0
        num_minibatch_updates = 0

        for _ in range(self.graph_iter_per_step): # PPO Epochs
            permutation_indices = np.random.permutation(num_samples_in_rollout)

            for start_idx in range(0, num_samples_in_rollout, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                mb_indices = permutation_indices[start_idx:end_idx]

                # 提取当前minibatch的数据
                # mb_states = all_old_states[mb_indices]

                # # 使用列表推导式对元组中的每个张量应用高级索引
                # minibatch_states_list = [state_part[mb_indices] for state_part in all_old_states]
                # # 将列表转换回元组
                # mb_states = tuple(minibatch_states_list)

                part_old_pyg_graph = Batch.from_data_list([all_old_states['pyg_graph'][temp_index] for temp_index in mb_indices]).cuda()
                part_old_current_idx = all_old_current_idx[mb_indices]
                part_old_action_idxes = all_old_action_idxes[mb_indices]
                part_old_action_mask = all_old_action_mask[mb_indices]
                mb_states = (part_old_pyg_graph, part_old_current_idx, part_old_action_idxes, part_old_action_mask)

                mb_actions = all_old_actions[mb_indices]
                mb_old_logprobs = all_old_logprobs[mb_indices]
                # mb_returns_G_t = rewards[mb_indices] # 当前minibatch的G_t
                mb_advantages = adv[mb_indices]
                mb_v_target = v_target[mb_indices]
        
                # 3b. 调用 train 函数处理这个minibatch
                actor_loss_this_minibatch, critic_loss_this_minibatch = self.train(
                    writer,
                    mb_states,
                    mb_actions,
                    mb_old_logprobs,
                    # mb_returns_G_t, # 将G_t作为"reward"参数传递给train函数
                    mb_advantages,
                    mb_v_target
                )
                accumulated_actor_loss += actor_loss_this_minibatch.mean().item()
                accumulated_critic_loss += critic_loss_this_minibatch.mean().item()
                num_minibatch_updates += 1
        
        actor_avg_loss_for_update = accumulated_actor_loss / num_minibatch_updates if num_minibatch_updates > 0 else 0
        critic_avg_loss_for_update = accumulated_critic_loss / num_minibatch_updates if num_minibatch_updates > 0 else 0
        
        
        self.actor_old.load_state_dict(self.actor.state_dict())
        # self.critic_old.load_state_dict(self.critic.state_dict())
        self.train_cnt += 1

        current_temp = torch.exp(self.actor.log_temperature).item()
        writer.add_scalar('Training/Policy/current_temp', current_temp, self.train_cnt)

        writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        writer.add_scalar('Training/Policy/lr_critic', self.critic_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        writer.add_scalar('Training/Policy/actor_avg_loss_per_update', actor_avg_loss_for_update, self.train_cnt)
        writer.add_scalar('Training/Policy/critic_avg_loss_per_update', critic_avg_loss_for_update, self.train_cnt)


    def train(self, writer,
            # 以下是minibatch数据：
            state_mb,         # (minibatch_size, state_dim...)
            action_mb,        # (minibatch_size, 1) - 动作索引
            old_logprob_mb,   # (minibatch_size, 1) - 旧策略的log_prob(a|s)
            # returns_G_t_mb,    # (minibatch_size,) - 这个minibatch的回报 G_t
            advantages_mb,
            v_target_mb
    ):
        self.actor.train()
        self.critic.train()

        # '''load data batch'''
        # state, action, reward, not_done, action_logprob = self.buffer.sample()
        # state = self.state_handler(state)
        # action = self.action_handler(action) # B,1 (选择到的action在action_space中的index)
        # reward = reward.unsqueeze(1).float().cuda() # B,1,1
        # not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
        # action_logprob = action_logprob.unsqueeze(1).float().cuda() # B,1,1
        

        '''actor'''
        origin_logprobs = self.actor(state_mb, self.args) # Batch, Num_Action # 用old_state得到对应的对数概率分布
        origin_logprobs = origin_logprobs.unsqueeze(-1).float().cuda() # B,N,1
        entropy = (origin_logprobs * origin_logprobs.exp()).sum(dim=1) # (B,1)
        logprobs = torch.gather(origin_logprobs, 1, action_mb.unsqueeze(-1).cuda()) # B,1,1
        ratios = torch.exp(logprobs - old_logprob_mb.detach().cuda()) # （B,1,1）

        surr1 = ratios * advantages_mb # （B,1,1）
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_mb # （B,1,1）
        actor_loss = -torch.min(surr1.squeeze(-1), surr2.squeeze(-1)) - self.ent_coef * entropy # Actor相关的损失

        # take gradient step
        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward() # 只计算影响actor的梯度
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


        origin_state_values = self.critic(state_mb, self.args) # Batch, Num_Action, 1
        state_values = origin_state_values.squeeze(2).cuda()
        critic_loss = self.MseLoss(state_values.squeeze(-1), v_target_mb.squeeze(-1))
        
        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.mean().backward() # 只计算影响critic的梯度
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return actor_loss, critic_loss


    # def train(self, writer,
    #         # 以下是minibatch数据：
    #         state_mb,         # (minibatch_size, state_dim...)
    #         action_mb,        # (minibatch_size, 1) - 动作索引
    #         old_logprob_mb,   # (minibatch_size, 1) - 旧策略的log_prob(a|s)
    #         returns_G_t_mb,    # (minibatch_size,) - 这个minibatch的回报 G_t
    #         advantages_mb
    # ):
    #     self.actor.train()
    #     self.critic.train()

    #     # '''load data batch'''
    #     # state, action, reward, not_done, action_logprob = self.buffer.sample()
    #     # state = self.state_handler(state)
    #     # action = self.action_handler(action) # B,1 (选择到的action在action_space中的index)
    #     # reward = reward.unsqueeze(1).float().cuda() # B,1,1
    #     # not_done = not_done.unsqueeze(1).float().cuda() # B,1,1
    #     # action_logprob = action_logprob.unsqueeze(1).float().cuda() # B,1,1
        

    #     '''actor'''
    #     origin_logprobs = self.actor(state_mb, self.args) # Batch, Num_Action # 用old_state得到对应的对数概率分布
    #     origin_logprobs = origin_logprobs.unsqueeze(-1).float().cuda() # B,N,1
    #     entropy = (origin_logprobs * origin_logprobs.exp()).sum(dim=1) # (B,1)
    #     logprobs = torch.gather(origin_logprobs, 1, action_mb.unsqueeze(-1)) # B,1,1
    #     ratios = torch.exp(logprobs - old_logprob_mb.detach()) # （B,1,1）

    #     surr1 = ratios * advantages_mb # （B,1,1）
    #     surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_mb # （B,1,1）

    #     origin_state_values = self.critic(state_mb, self.args) # Batch, Num_Action, 1
    #     # state_values = torch.gather(origin_state_values, 1, action_mb.unsqueeze(-1)) # B,1,1(选择action对应的q值)
    #     state_values = origin_state_values.squeeze(2)

    #     policy_loss = -torch.min(surr1.squeeze(-1), surr2.squeeze(-1))
    #     critic_loss = self.MseLoss(state_values.squeeze(-1), returns_G_t_mb.squeeze(-1))
    #     loss = policy_loss + self.vf_coef*critic_loss - self.ent_coef*entropy

    #     # take gradient step
    #     # 优化 Actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss_for_backward = policy_loss - self.ent_coef * entropy # Actor相关的损失
    #     actor_loss_for_backward.mean().backward() # 只计算影响actor的梯度
    #     self.actor_optimizer.step()

    #     # 优化 Critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss_for_backward = self.vf_coef * critic_loss # Critic相关的损失
    #     critic_loss_for_backward.mean().backward() # 只计算影响critic的梯度
    #     self.critic_optimizer.step()

    #     # loss.mean().backward()
    #     # self.actor_optimizer.step()
    #     return loss
        
        # writer.add_scalar('Training/Policy/lr_actor', self.actor_optimizer.state_dict()['param_groups'][0]['lr'], self.train_cnt)
        # writer.add_scalar('Training/Policy/loss', loss.item(), self.train_cnt)

        # if self.train_step % self.lr_scheduler_interval == 0:
        #     self.scheduler_actor.step()
                    
    def save(self, dir_path):
        save_models(self.critic, self.critic_optimizer, "critic", dir_path)
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
        self.actor.load_state_dict(checkpoint, strict=False)
        self.actor_old.load_state_dict(checkpoint, strict=False)

        # 筛选出Encoder的参数
        # 筛选逻辑: 保留所有键名以 "encoder." 开头的项
        encoder_weights = {key: value for key, value in checkpoint.items() if key.startswith("pre.")}

        # 使用 strict=False 将筛选后的权重加载到Critic模型中
        # 这会只更新encoder部分的权重，而critic_head部分保持不变
        self.critic.load_state_dict(encoder_weights, strict=False)




