import numpy as np
from navigation.habitat_action import HabitatAction
from env_tools.arguments import args

from navigation.sub_goal_reach import SubgoalReach

class Evaluate:
    success_num = 0 # sr
    spl_per_episode = 0
    spl_ls = []
    spl_mean = 0
    
    ne_per_episode = 0
    ne_ls = []
    ne_mean = 0

    state_dict = {"achieved": 1, "exceed": -1, "Failed_Plan": -2, "empty": -3, "block": -4, "EXCEED_RL": -5}

    # val
    all_front_steps = 0

    # train
    real_episode_num_in_train = 1
    empty_num = 0
    block_num = 0
    failed_plan_num = 0
    exceed_rl_num = 0


    max_front_steps_per_rl_step = 200

    spl_per_episode_general = 0
    spl_per_episode_limit = 0

    sr_buffer_general_20 = []
    sr_buffer_limit_20 = []

    spl_buffer_general_20 = []
    spl_buffer_limit_20 = []

    @staticmethod
    def reset():
        Evaluate.success_num = 0 # sr
        Evaluate.spl_per_episode = 0
        Evaluate.spl_ls = []
        Evaluate.spl_mean = 0
        Evaluate.ne_per_episode = 0
        Evaluate.ne_ls = []
        Evaluate.ne_mean = 0

        Evaluate.all_front_steps = 0


    @staticmethod
    def evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, graph_train=False, rl_graph=None, policy=None):
        # 1. 成功到达
        # 0. 到达错误的label goal
        # -1. 超过最大步长
        # -2. rrt规划失败
        # -3. action space为空
        # -4. 卡住（只记录intention node的卡住）
        
        habitat_metric = habitat_env.get_metrics()
        print("habitat_metric:", habitat_metric)
        
        
        if(graph_train==True):
            if(achieved_result=="exceed" or achieved_result=="empty" or achieved_result=="block" or achieved_result=="Failed_Plan"):
                if(achieved_result=="empty"):
                    Evaluate.empty_num += 1
                elif(achieved_result=="block"):
                    Evaluate.block_num += 1
                elif(achieved_result=="Failed_Plan"):
                    Evaluate.failed_plan_num += 1
                    
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                return "episode_stop" # 结束当前episode, 开始下一个episode
            
            elif(achieved_result=="achieved"):
                if(action_node.node_type=="frontier_node"):
                    reward_per_rl_step = (HabitatAction.front_steps-SubgoalReach.init_front_steps)*(-1)/Evaluate.max_front_steps_per_rl_step+0
                    rl_graph.data['arrive'] = False
                    HabitatAction.reward_per_episode += reward_per_rl_step

                elif(action_node.node_type=="intention_node"):
                    distance_to_goal = habitat_metric['distance_to_goal']
                    if(distance_to_goal<=1.0):
                        reward_per_rl_step = (HabitatAction.front_steps-SubgoalReach.init_front_steps)*(-1)/Evaluate.max_front_steps_per_rl_step+40
                    else:
                        reward_per_rl_step = (HabitatAction.front_steps-SubgoalReach.init_front_steps)*(-1)/Evaluate.max_front_steps_per_rl_step-40
                    rl_graph.data['arrive'] = True
                    HabitatAction.reward_per_episode += reward_per_rl_step
                    writer.add_scalar('Result/reward_per_episode', HabitatAction.reward_per_episode, Evaluate.real_episode_num_in_train)

                    # =====> sr & spl <=====
                    # =====> spl_per_episode <=====
                    Evaluate.spl_per_episode_general = habitat_metric['spl']
                    if(HabitatAction.count_steps<=args.max_steps):
                        Evaluate.spl_per_episode_limit = habitat_metric['spl']
                    else:
                        Evaluate.spl_per_episode_limit = 0

                    # =====> spl_buffer <=====
                    Evaluate.spl_buffer_general_20.append(Evaluate.spl_per_episode_general)
                    Evaluate.spl_buffer_limit_20.append(Evaluate.spl_per_episode_limit)

                    # =====> sr_buffer <=====
                    Evaluate.sr_buffer_general_20.append(habitat_metric['success'])
                    if(Evaluate.spl_per_episode_limit>0):
                        Evaluate.sr_buffer_limit_20.append(1)
                    else:
                        Evaluate.sr_buffer_limit_20.append(0)

                    writer.add_scalar('Result/spl_per_episode_general', Evaluate.spl_per_episode_general, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/spl_per_episode_limit', Evaluate.spl_per_episode_limit, Evaluate.real_episode_num_in_train)
                    if(len(Evaluate.spl_buffer_general_20)>=20):
                        if(len(Evaluate.spl_buffer_general_20)>20):
                            Evaluate.spl_buffer_general_20.pop(0)
                            Evaluate.spl_buffer_limit_20.pop(0)
                            Evaluate.sr_buffer_general_20.pop(0)
                            Evaluate.sr_buffer_limit_20.pop(0)
                    
                        writer.add_scalar('Result/spl_average_general', sum(Evaluate.spl_buffer_general_20)/20, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/spl_average_limit', sum(Evaluate.spl_buffer_limit_20)/20, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/sr_average_general', sum(Evaluate.sr_buffer_general_20)/20, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/sr_average_limit', sum(Evaluate.sr_buffer_limit_20)/20, Evaluate.real_episode_num_in_train)
                    Evaluate.real_episode_num_in_train += 1

                rl_graph.data['reward'] = reward_per_rl_step
                writer.add_scalar('Result/reward_per_rl_step', reward_per_rl_step, policy.train_step+1)
                
                
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                
                if(action_node.node_type=="frontier_node"):
                    return "next_rl_step"
                elif(action_node.node_type=="intention_node"):
                    return "episode_stop"
            
            elif(achieved_result=="EXCEED_RL"):
                reward_per_rl_step = (HabitatAction.front_steps-SubgoalReach.init_front_steps)*(-1)/Evaluate.max_front_steps_per_rl_step+0
                rl_graph.data['arrive'] = False
                HabitatAction.reward_per_episode += reward_per_rl_step
                writer.add_scalar('Result/reward_per_episode', HabitatAction.reward_per_episode, Evaluate.real_episode_num_in_train)

                # =====> sr & spl <=====
                # =====> spl_per_episode <=====
                Evaluate.spl_per_episode_general = 0
                Evaluate.spl_per_episode_limit = 0

                # =====> spl_buffer <=====
                Evaluate.spl_buffer_general_20.append(Evaluate.spl_per_episode_general)
                Evaluate.spl_buffer_limit_20.append(Evaluate.spl_per_episode_limit)

                # =====> sr_buffer <=====
                Evaluate.sr_buffer_general_20.append(0)
                Evaluate.sr_buffer_limit_20.append(0)

                writer.add_scalar('Result/spl_per_episode_general', Evaluate.spl_per_episode_general, Evaluate.real_episode_num_in_train)
                writer.add_scalar('Result/spl_per_episode_limit', Evaluate.spl_per_episode_limit, Evaluate.real_episode_num_in_train)
                if(len(Evaluate.spl_buffer_general_20)>=20):
                    if(len(Evaluate.spl_buffer_general_20)>20):
                        Evaluate.spl_buffer_general_20.pop(0)
                        Evaluate.spl_buffer_limit_20.pop(0)
                        Evaluate.sr_buffer_general_20.pop(0)
                        Evaluate.sr_buffer_limit_20.pop(0)
                
                    writer.add_scalar('Result/spl_average_general', sum(Evaluate.spl_buffer_general_20)/20, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/spl_average_limit', sum(Evaluate.spl_buffer_limit_20)/20, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/sr_average_general', sum(Evaluate.sr_buffer_general_20)/20, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/sr_average_limit', sum(Evaluate.sr_buffer_limit_20)/20, Evaluate.real_episode_num_in_train)
                Evaluate.real_episode_num_in_train += 1

                rl_graph.data['reward'] = reward_per_rl_step
                writer.add_scalar('Result/reward_per_rl_step', reward_per_rl_step, policy.train_step+1)

                Evaluate.exceed_rl_num += 1
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                return "episode_stop"


        else: # 处于测试阶段
            Evaluate.all_front_steps += HabitatAction.front_steps
            if(achieved_result=="exceed" or achieved_result=="empty"):
                if(habitat_metric['success']>0):
                    Evaluate.success_num += 1
                Evaluate.spl_per_episode = habitat_metric['spl']
                Evaluate.spl_ls.append(Evaluate.spl_per_episode)
                Evaluate.spl_mean = np.mean(Evaluate.spl_ls)

                Evaluate.ne_per_episode = habitat_metric['distance_to_goal']
                Evaluate.ne_ls.append(Evaluate.ne_per_episode)
                Evaluate.ne_mean = np.mean(Evaluate.ne_ls)
                
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/walk_path_meter', HabitatAction.walk_path_meter, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                writer.add_scalar('Result/success_num', Evaluate.success_num, index_in_episodes+1)
                writer.add_scalar('Result/spl_per_episode', Evaluate.spl_per_episode, index_in_episodes+1)
                writer.add_scalar('Result/spl_mean', Evaluate.spl_mean, index_in_episodes+1)
                writer.add_scalar('Result/ne_per_episode', Evaluate.ne_per_episode, index_in_episodes+1)
                writer.add_scalar('Result/ne_mean', Evaluate.ne_mean, index_in_episodes+1)
                return "episode_stop" # 结束当前episode, 开始下一个episode
                
            elif(achieved_result=="achieved" or achieved_result=="block" or achieved_result=="Failed_Plan"):
                if(action_node.node_type=="frontier_node"):
                    return "next_rl_step" # 继续选择下一个action

                elif(action_node.node_type=="intention_node"):
                    if(habitat_metric['success']>0):
                        Evaluate.success_num += 1
                    Evaluate.spl_per_episode = habitat_metric['spl']
                    Evaluate.spl_ls.append(Evaluate.spl_per_episode)
                    Evaluate.spl_mean = np.mean(Evaluate.spl_ls)

                    Evaluate.ne_per_episode = habitat_metric['distance_to_goal']
                    Evaluate.ne_ls.append(Evaluate.ne_per_episode)
                    Evaluate.ne_mean = np.mean(Evaluate.ne_ls)
                    
                    writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                    writer.add_scalar('Result/walk_path_meter', HabitatAction.walk_path_meter, index_in_episodes+1)
                    writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                    writer.add_scalar('Result/success_num', Evaluate.success_num, index_in_episodes+1)
                    writer.add_scalar('Result/spl_per_episode', Evaluate.spl_per_episode, index_in_episodes+1)
                    writer.add_scalar('Result/spl_mean', Evaluate.spl_mean, index_in_episodes+1)
                    writer.add_scalar('Result/ne_per_episode', Evaluate.ne_per_episode, index_in_episodes+1)
                    writer.add_scalar('Result/ne_mean', Evaluate.ne_mean, index_in_episodes+1)
                    return "episode_stop" # 结束当前episode, 开始下一个episode
                