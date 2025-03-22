import numpy as np
from navigation.habitat_action import HabitatAction
from env_tools.arguments import args

from navigation.sub_goal_reach import SubgoalReach
from graph.tools import get_absolute_pos

from perception.arguments import args as perception_args

from graph.tools import get_current_world_pos, find_node_path

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
    all_count_steps = 0
    all_dis_shape = 0

    # train
    real_episode_num_in_train = 1
    empty_num = 0
    block_num = 0
    failed_plan_num = 0
    exceed_rl_num = 0

    reward_false_num = 0

    # ============> front_step_origin <==============
    max_front_steps_per_rl_step = 200
    # ============> front_step_origin <==============
    max_count_steps_per_rl_step = 500

    spl_per_episode_general = 0
    spl_per_episode_limit = 0

    sr_buffer_general_30 = []
    sr_buffer_limit_30 = []

    spl_buffer_general_30 = []
    spl_buffer_limit_30 = []

    

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
        Evaluate.all_count_steps = 0
        Evaluate.all_dis_shape = 0

    @staticmethod
    def get_topo_walk_dis(action_node, topo_graph):
        topo_walk_dis = 0
        start_explored_node = SubgoalReach.init_explored_node
        end_explored_node = action_node.parent_node
        start_end_node_path = find_node_path(start_explored_node, end_explored_node, topo_graph.explored_nodes)
        for temp_index, temp_node in enumerate(start_end_node_path):
            if(temp_index==0):
                continue
            topo_walk_dis += ((start_end_node_path[temp_index].world_cx-start_end_node_path[temp_index-1].world_cx)**2+(start_end_node_path[temp_index].world_cy-start_end_node_path[temp_index-1].world_cy)**2)**0.5
        topo_walk_dis += (SubgoalReach.init_rela_cx**2+SubgoalReach.init_rela_cy**2)**0.5
        topo_walk_dis += (action_node.rela_cx**2+action_node.rela_cy**2)**0.5
        return topo_walk_dis


    @staticmethod
    def evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, graph_train=False, rl_graph=None, policy=None, topo_graph=None, scene_area=None):
        # 1. 成功到达
        # 0. 到达错误的label goal
        # -1. 超过最大步长
        # -2. rrt规划失败
        # -3. action space为空
        # -4. 卡住（只记录intention node的卡住）
        
        habitat_metric = habitat_env.get_metrics()
        print("habitat_metric:", habitat_metric)
        
        
        if(graph_train==True):
            distance_to_goal = habitat_metric['distance_to_goal']
            if(achieved_result=="exceed" or achieved_result=="empty" or achieved_result=="block" or achieved_result=="Failed_Plan"):
                if(achieved_result=="empty"):
                    Evaluate.empty_num += 1
                elif(achieved_result=="block"):
                    Evaluate.block_num += 1
                elif(achieved_result=="Failed_Plan"):
                    Evaluate.failed_plan_num += 1
                    
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/reward_false_num', Evaluate.reward_false_num, index_in_episodes+1)

                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                return "episode_stop" # 结束当前episode, 开始下一个episode
            
            elif(achieved_result=="achieved"):
                # =============> reward_revise <=============
                world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                if(now_action_dis>1) or ((HabitatAction.front_steps-SubgoalReach.init_front_steps)==0):                    
                    Evaluate.reward_false_num += 1
                    writer.add_scalar('Result/episode_state', -6, index_in_episodes+1)
                    writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                    writer.add_scalar('Result/reward_false_num', Evaluate.reward_false_num, index_in_episodes+1)

                    writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                    writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                    writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                    writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                    return "false_reward"
                # =============> reward_revise <=============
                
                
                if(action_node.node_type=="frontier_node"):
                    topo_walk_dis = Evaluate.get_topo_walk_dis(action_node, topo_graph)
                    
                    init_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                    HabitatAction.get_all_map_loc(topo_graph)
                    now_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                    delta_area_percentage = ((now_all_map_loc_num-init_all_map_loc_num)/100)/scene_area

                    init_intention_num = len(HabitatAction.real_intention_nodes)
                    HabitatAction.get_all_see_intention(topo_graph, rl_graph)

                    if(len(HabitatAction.real_intention_nodes)==0): # 没有看到真正的intention_node
                        reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+delta_area_percentage*20
                    else: # 看到了真正的intention
                        if(init_intention_num==0): # 表示第一次看到真正的intention
                            reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+10
                        else:
                            reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+0

                    rl_graph.data['arrive'] = False
                    HabitatAction.reward_per_episode += reward_per_rl_step

                elif(action_node.node_type=="intention_node"):
                    topo_walk_dis = Evaluate.get_topo_walk_dis(action_node, topo_graph)

                    init_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                    HabitatAction.get_all_map_loc(topo_graph)
                    now_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                    delta_area_percentage = ((now_all_map_loc_num-init_all_map_loc_num)/100)/scene_area

                    init_intention_num = len(HabitatAction.real_intention_nodes)
                    HabitatAction.get_all_see_intention(topo_graph, rl_graph)

                    
                    if(distance_to_goal<=1.0): # 如果最终成功，则认为一定看到了真正的intention_node
                        if(init_intention_num==0): 
                            reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+10+40
                        else:
                            reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+40
                    else: # 如果最终失败，则不一定看到真正的intention_node
                        if(len(HabitatAction.real_intention_nodes)==0): # 没有看到真正的intention_node
                            reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+delta_area_percentage*20
                        else:   
                            if(init_intention_num==0):      
                                reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+10
                            else:
                                reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+0
                    
                    rl_graph.data['arrive'] = True
                    HabitatAction.reward_per_episode += reward_per_rl_step
                    writer.add_scalar('Result/reward_per_episode', HabitatAction.reward_per_episode, Evaluate.real_episode_num_in_train)

                    # =====> sr & spl <=====
                    # =====> spl_per_episode <=====
                    Evaluate.spl_per_episode_general = habitat_metric['spl']
                    if(HabitatAction.count_steps<=500):
                        Evaluate.spl_per_episode_limit = habitat_metric['spl']
                    else:
                        Evaluate.spl_per_episode_limit = 0

                    # =====> spl_buffer <=====
                    Evaluate.spl_buffer_general_30.append(Evaluate.spl_per_episode_general)
                    Evaluate.spl_buffer_limit_30.append(Evaluate.spl_per_episode_limit)

                    # =====> sr_buffer <=====
                    Evaluate.sr_buffer_general_30.append(habitat_metric['success'])
                    if(Evaluate.spl_per_episode_limit>0):
                        Evaluate.sr_buffer_limit_30.append(1)
                    else:
                        Evaluate.sr_buffer_limit_30.append(0)

                    writer.add_scalar('Result/spl_per_episode_general', Evaluate.spl_per_episode_general, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/spl_per_episode_limit', Evaluate.spl_per_episode_limit, Evaluate.real_episode_num_in_train)
                    if(len(Evaluate.spl_buffer_general_30)>=30):
                        if(len(Evaluate.spl_buffer_general_30)>30):
                            Evaluate.spl_buffer_general_30.pop(0)
                            Evaluate.spl_buffer_limit_30.pop(0)
                            Evaluate.sr_buffer_general_30.pop(0)
                            Evaluate.sr_buffer_limit_30.pop(0)
                    
                        writer.add_scalar('Result/spl_average_general', sum(Evaluate.spl_buffer_general_30)/30, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/spl_average_limit', sum(Evaluate.spl_buffer_limit_30)/30, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/sr_average_general', sum(Evaluate.sr_buffer_general_30)/30, Evaluate.real_episode_num_in_train)
                        writer.add_scalar('Result/sr_average_limit', sum(Evaluate.sr_buffer_limit_30)/30, Evaluate.real_episode_num_in_train)
                    Evaluate.real_episode_num_in_train += 1

                rl_graph.data['reward'] = reward_per_rl_step
                writer.add_scalar('Result/reward_per_rl_step', reward_per_rl_step, policy.train_step+1)

                writer.add_scalar('Result/topo_walk_dis', topo_walk_dis, policy.train_step+1)
                writer.add_scalar('Result/real_walk_dis', (HabitatAction.front_steps-SubgoalReach.init_front_steps)*0.25, policy.train_step+1)
                
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/reward_false_num', Evaluate.reward_false_num, index_in_episodes+1)

                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                
                if(action_node.node_type=="frontier_node"):
                    return "next_rl_step"
                elif(action_node.node_type=="intention_node"):
                    return "episode_stop"
            
            elif(achieved_result=="EXCEED_RL"):
                # =============> reward_revise <=============
                world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                if(now_action_dis>1) or ((HabitatAction.front_steps-SubgoalReach.init_front_steps)==0):                    
                    Evaluate.reward_false_num += 1
                    writer.add_scalar('Result/episode_state', -6, index_in_episodes+1)
                    writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                    writer.add_scalar('Result/reward_false_num', Evaluate.reward_false_num, index_in_episodes+1)

                    writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                    writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                    writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                    writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                    return "false_reward"
                # =============> reward_revise <=============
                topo_walk_dis = Evaluate.get_topo_walk_dis(action_node, topo_graph)
                    
                init_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                HabitatAction.get_all_map_loc(topo_graph)
                now_all_map_loc_num = HabitatAction.init_all_map_loc.shape[0]
                delta_area_percentage = ((now_all_map_loc_num-init_all_map_loc_num)/100)/scene_area

                init_intention_num = len(HabitatAction.real_intention_nodes)
                HabitatAction.get_all_see_intention(topo_graph, rl_graph)

                if(len(HabitatAction.real_intention_nodes)==0): # 没有看到intention_node
                    reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+delta_area_percentage*20
                else: # 看到了真正的intention
                    if(init_intention_num==0): # 表示第一次看到了真正的intention
                        reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+10
                    else:
                        reward_per_rl_step = (topo_walk_dis/0.25)*(-1)/12.5+0
                
                rl_graph.data['arrive'] = False
                HabitatAction.reward_per_episode += reward_per_rl_step
                writer.add_scalar('Result/reward_per_episode', HabitatAction.reward_per_episode, Evaluate.real_episode_num_in_train)

                # =====> sr & spl <=====
                # =====> spl_per_episode <=====
                Evaluate.spl_per_episode_general = 0
                Evaluate.spl_per_episode_limit = 0

                # =====> spl_buffer <=====
                Evaluate.spl_buffer_general_30.append(Evaluate.spl_per_episode_general)
                Evaluate.spl_buffer_limit_30.append(Evaluate.spl_per_episode_limit)

                # =====> sr_buffer <=====
                Evaluate.sr_buffer_general_30.append(0)
                Evaluate.sr_buffer_limit_30.append(0)

                writer.add_scalar('Result/spl_per_episode_general', Evaluate.spl_per_episode_general, Evaluate.real_episode_num_in_train)
                writer.add_scalar('Result/spl_per_episode_limit', Evaluate.spl_per_episode_limit, Evaluate.real_episode_num_in_train)
                if(len(Evaluate.spl_buffer_general_30)>=30):
                    if(len(Evaluate.spl_buffer_general_30)>30):
                        Evaluate.spl_buffer_general_30.pop(0)
                        Evaluate.spl_buffer_limit_30.pop(0)
                        Evaluate.sr_buffer_general_30.pop(0)
                        Evaluate.sr_buffer_limit_30.pop(0)
                
                    writer.add_scalar('Result/spl_average_general', sum(Evaluate.spl_buffer_general_30)/30, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/spl_average_limit', sum(Evaluate.spl_buffer_limit_30)/30, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/sr_average_general', sum(Evaluate.sr_buffer_general_30)/30, Evaluate.real_episode_num_in_train)
                    writer.add_scalar('Result/sr_average_limit', sum(Evaluate.sr_buffer_limit_30)/30, Evaluate.real_episode_num_in_train)
                Evaluate.real_episode_num_in_train += 1

                rl_graph.data['reward'] = reward_per_rl_step
                writer.add_scalar('Result/reward_per_rl_step', reward_per_rl_step, policy.train_step+1)

                writer.add_scalar('Result/topo_walk_dis', topo_walk_dis, policy.train_step+1)
                writer.add_scalar('Result/real_walk_dis', (HabitatAction.front_steps-SubgoalReach.init_front_steps)*0.25, policy.train_step+1)

                Evaluate.exceed_rl_num += 1
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/empty_num', Evaluate.empty_num, index_in_episodes+1)
                writer.add_scalar('Result/reward_false_num', Evaluate.reward_false_num, index_in_episodes+1)

                writer.add_scalar('Result/block_num', Evaluate.block_num, index_in_episodes+1)
                writer.add_scalar('Result/failed_plan_num', Evaluate.failed_plan_num, index_in_episodes+1)
                writer.add_scalar('Result/exceed_rl_num', Evaluate.exceed_rl_num, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)

                return "episode_stop"


        else: # 处于测试阶段
            Evaluate.all_front_steps += HabitatAction.front_steps
            Evaluate.all_count_steps += HabitatAction.count_steps
            Evaluate.all_dis_shape += (HabitatAction.this_episode_short_dis-habitat_metric['distance_to_goal'])
            if(achieved_result=="exceed" or achieved_result=="empty"):
                if(habitat_metric['success']>0):
                    Evaluate.success_num += 1
                    writer.add_scalar('Simulator/ratio_state', 1, index_in_episodes+1)
                elif(achieved_result=="exceed"):
                    writer.add_scalar('Simulator/ratio_state', -1, index_in_episodes+1)
                elif(achieved_result=="empty"):
                    writer.add_scalar('Simulator/ratio_state', -3, index_in_episodes+1)

                # # =================> evaluate_revise <=================
                # if(achieved_result=="exceed"):
                #     writer.add_scalar('Simulator/ratio_state', -1, index_in_episodes+1)
                # elif(achieved_result=="empty"):
                #     writer.add_scalar('Simulator/ratio_state', -3, index_in_episodes+1)
                # # =================> evaluate_revise <=================


                Evaluate.spl_per_episode = habitat_metric['spl']
                # # =================> evaluate_revise <=================
                # Evaluate.spl_per_episode = 0
                # # =================> evaluate_revise <=================
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
                        writer.add_scalar('Simulator/ratio_state', 1, index_in_episodes+1)
                    elif(achieved_result=="block"):
                        writer.add_scalar('Simulator/ratio_state', -4, index_in_episodes+1)
                    elif(achieved_result=="Failed_Plan"):
                        writer.add_scalar('Simulator/ratio_state', -2, index_in_episodes+1)
                    else:
                        world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                        now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                        
                        if(now_action_dis>1):
                            writer.add_scalar('Simulator/ratio_state', -5, index_in_episodes+1) # 底层控制器失败
                        else:
                            writer.add_scalar('Simulator/ratio_state', 0, index_in_episodes+1) # 到达错误的intention_node

                    # # =================> evaluate_revise <=================
                    # if(achieved_result=="block"):
                    #     Evaluate.spl_per_episode = 0
                    #     writer.add_scalar('Simulator/ratio_state', -4, index_in_episodes+1)
                    # elif(achieved_result=="Failed_Plan"):
                    #     Evaluate.spl_per_episode = 0
                    #     writer.add_scalar('Simulator/ratio_state', -2, index_in_episodes+1)
                    # else:
                    #     if(habitat_metric['success']>0):
                    #         Evaluate.success_num += 1
                    #         Evaluate.spl_per_episode = habitat_metric['spl']
                    #         writer.add_scalar('Simulator/ratio_state', 1, index_in_episodes+1)
                    #     else:
                    #         Evaluate.spl_per_episode = 0
                    #         world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                    #         now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                            
                    #         if(now_action_dis>1):
                    #             writer.add_scalar('Simulator/ratio_state', -5, index_in_episodes+1) # 底层控制器失败
                    #         else:
                    #             writer.add_scalar('Simulator/ratio_state', 0, index_in_episodes+1) # 到达错误的intention_node
                    # # =================> evaluate_revise <=================


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