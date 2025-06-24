import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import random
random.seed(233)

import cv2
import copy
import habitat
import argparse
import datetime
import numpy as np
np.random.seed(233)

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env_tools.arguments import args as env_args
from env_tools.data_utils import hm3d_config, habitat_camera_intrinsic
from env_tools.evaluate_utils import Evaluate

from policy.rl_algorithms.arguments import args as rl_args
from system_utils import process_info
from policy.tools.utils import init_RL, system_info
from policy.rl_algorithms.rl_graph import RL_Graph

from perception.tools import fix_depth, get_rgb_image_ls, get_gt_image_ls
from perception.arguments import args as perception_args
from graph.graph_utils import GraphMap
from graph.node_utils import Node
from graph.tools import get_current_world_pos, get_min_goal_loc, get_absolute_pos

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

# from perception.intention_utils_rcnn import object_detect
from perception.intention_utils_gt import object_detect_gt
# from perception.intention_utils_gt_other import object_detect_gt_other


if __name__=="__main__":
    env_args.task_stage = "train"
    env_args.graph_train = True
    env_args.root = "/home/zhaishichao/Data/VLN"
    env_args.model_file_name = "Models_train_PPO"
    env_args.graph_pre_model = 683
    # env_args.graph_pre_model = 0

    train_note = "_multi_check_ppo_large_punish_reward" # 注释当前训练处于什么阶段
    # train_note = "_multi_check_ppo_large_punish_reward_from_scratch" # 注释当前训练处于什么阶段

    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    env_args.logger_file_name = "./log_files_train_ppo_ring_il_pretrain/log_"+date_time+train_note
    env_args.graph_episode_num = 80000
    env_args.success_distance = 1.0
    env_args.max_steps = 10000

    # only_train
    env_args.graph_model_save_frequency = 1
    # env_args.graph_batch_size = 64
    env_args.graph_episode_length = 40 # max train rl_steps for per episode
    

    rl_args.score_top_k = 50
    rl_args.graph_node_feature_dim = 102 # 分数序列+距离序列+frontier_score+类型
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = False
    # rl_args.update_timestep = 200      # update policy every n timesteps
    rl_args.update_timestep = 512      # update policy every n timesteps
    rl_args.minibatch_size = 64

    # rl_args.update_timestep = 4      # update policy every n timesteps
    # rl_args.minibatch_size = 2

    # new_revise
    env_args.graph_iter_per_step = 2 # 每个训练2次
    # env_args.graph_iter_per_step = 4 # 每个训练4次
    rl_args.lr_tune = 0.5e-3
    rl_args.random_exploration_length = 0 # 不需要random_walk
    
    # only_train
    rl_args.save_buffer_data_path = "buffer_data/{}/".format(date_time)
    rl_args.load_buffer_data_cnt = -1
    rl_args.load_buffer_data_path = ""

    writer = SummaryWriter(env_args.logger_file_name)

    habitat_config = hm3d_config(stage=env_args.task_stage, episodes=1, max_steps=env_args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)
    perception_args.intrinsic_matrix = habitat_camera_intrinsic(config=habitat_config)

    experiment_details = 'graph_'  + rl_args.graph_task + '_' + rl_args.graph_action_space + \
        '_'+ rl_args.graph_encoder+"_"+date_time+train_note
    init_free_memory, init_process_memory = process_info()
    policy = init_RL(env_args, rl_args, experiment_details, writer=writer)

    # =====> select episodes <=====
    id_dict = {
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb":["tv_monitor", "bed", "sofa", "chair", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb":["tv_monitor", "toilet", "chair", "plant", "sofa"], 
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb":["sofa", "bed", "plant", "tv_monitor", "toilet"], 
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb":["bed", "toilet", "chair", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":["bed", "tv_monitor"],

        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":["bed", "tv_monitor", "toilet", "sofa", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb":["bed", "tv_monitor"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb":["tv_monitor", "toilet", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb":["bed", "toilet", "sofa", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb":["bed", "toilet", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb":["chair", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":["bed", "toilet", "chair"],
    }

    area_dict = {
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb": 129.99,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb": 220.18,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb": 171.28,  
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb": 110.49 ,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb": 259.68,

        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb": 212.73,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb": 120.95,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb": 420.58,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb": 451.53,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb": 881.54,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb": 312.67,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb": 225.5,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb": 179.34,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb": 142.1,
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb": 1059.28,
    }

    selected_episodes = []
    for index, temp_episode in enumerate(habitat_env.episodes):
        if(temp_episode.scene_id in id_dict.keys()):
            if(temp_episode.object_category in id_dict[temp_episode.scene_id]):
                selected_episodes.append(temp_episode)
    random.shuffle(selected_episodes)
    # =====> select episodes <=====


    reward_per_update = 0
    total_model_index = 1
    valid_episode_index = 1

    for index_in_episodes in tqdm(range(env_args.graph_episode_num)):
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        habitat_env.episodes = [selected_episodes[index_in_episodes]]
        writer.add_scalar('Scene/scene_id', list(id_dict.keys()).index(habitat_env.episodes[0].scene_id), index_in_episodes+1)
        print("=====> scene_id <=====", habitat_env.episodes[0].scene_id)
        try:
            observations = habitat_env.reset()
        except:
            continue

        object_goal = env_args.object_ls[observations["objectgoal"][0]]
        print("=====> object_goal <=====", object_goal)
        HabitatAction.reset(habitat_env, object_goal, env_args.graph_train) 
        habitat_metric = habitat_env.get_metrics()
        # topo_graph_init
        topo_graph = GraphMap(habitat_env=habitat_env)
        topo_graph.set_current_pos(rela_cx=0.0, rela_cy=0.0, rela_turn=0.0)
        graph_update_flag = topo_graph.update()
        
        for i in range(12):
            # get sensor data: depth, 2d_laser
            depth = fix_depth(observations["depth"])
            topo_graph.get_laser_result(depth)
            topo_graph.current_node.update_occupancy(topo_graph.laser_2d_filtered, topo_graph.laser_2d_filtered_angle, topo_graph.pixel_y_2d_filtered, np.array([topo_graph.rela_cx, topo_graph.rela_cy]), topo_graph.rela_turn)
            topo_graph.update_graph_frontier()

            rgb_image_ls = get_rgb_image_ls(habitat_env)
            gt_image_ls = get_gt_image_ls(habitat_env)
            # detect_res_pos_dict = object_detect(rgb_image_ls, depth, object_goal)
            # topo_graph.add_intention(detect_res_pos_dict)
            
            detect_res_pos_dict = object_detect_gt(gt_image_ls, depth, object_goal, HabitatAction.object_id_num_ls)
            topo_graph.add_intention_gt(detect_res_pos_dict, rgb_image_ls, object_goal)

            # other_res_pos_dict = object_detect_gt_other(gt_image_ls, depth, HabitatAction.other_object_id_num_ls)
            # topo_graph.add_other_intention(other_res_pos_dict)

            # 底层仿真器动作执行
            habitat_action = HabitatAction.set_habitat_action("r", topo_graph)
            observations = habitat_env.step(habitat_action)
            topo_graph.obs = observations
            
            # 用于录制视频
            if(env_args.is_vis==True):
                save_mp4(occu_writer, video_writer, map_writer, gt_writer, habitat_env, topo_graph, rl_graph, action_node=None, object_goal=object_goal)
        
        # rl_graph_update
        rl_graph.update(topo_graph)
        current_state = copy.deepcopy(rl_graph.data['state'])


        # 获得初始all_map_loc和初始intention_node的个数
        HabitatAction.get_all_map_loc(topo_graph)
        HabitatAction.get_all_see_intention(topo_graph, rl_graph)

        # 用于手动调试
        if(env_args.is_auto==False):
            cv2.imshow("rgb", rgb_image_ls[0])
            cv2.imshow("depth", observations["depth"])
            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occu_for_show", occu_for_show)

        episode_total_reward = 0
        
        while True:   
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                polict_action, policy_acton_idx, action_logprob = policy.select_action(writer, rl_graph.data['state'], if_train=env_args.graph_train)
                action_node = rl_graph.all_nodes[polict_action-1]
                print("=====> real_action_selection <=====")
            else:
                ghost_patch_res = topo_graph.ghost_patch(habitat_env, object_goal)
                rl_graph.update(topo_graph)
                if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0) and (ghost_patch_res=="ok"):
                    print("=====> ghost_patch <=====")
                    current_state = copy.deepcopy(rl_graph.data['state']) # 1106最新修改

                    polict_action, policy_acton_idx, action_logprob = policy.select_action(writer, rl_graph.data['state'], if_train=env_args.graph_train)
                    action_node = rl_graph.all_nodes[polict_action-1]
                    # 获得初始all_map_loc和初始intention_node的个数
                    HabitatAction.get_all_map_loc(topo_graph)
                    HabitatAction.get_all_see_intention(topo_graph, rl_graph)
                
                else:
                    # action_space为空，结束当前episode
                    if not habitat_env.episode_over: # 没有超过1w步的最大步长
                        habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                        observations = habitat_env.step(habitat_action)
                        achieved_result = "empty"
                    else:
                        achieved_result = "exceed"
                    Evaluate.evaluate(writer, achieved_result=achieved_result, habitat_env=habitat_env, action_node=None, index_in_episodes=index_in_episodes, graph_train=env_args.graph_train, rl_graph=rl_graph, policy=policy, topo_graph=topo_graph, scene_area=area_dict[habitat_env.current_episode.scene_id])
                    
                    if(len(policy.buffer.reward_arrives)!=0):
                        policy.delete_buffer()
                    break

            
            if(action_node.intention_type==2):
                if not habitat_env.episode_over:
                    habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                    observations = habitat_env.step(habitat_action)
                    achieved_result = "achieved"
                else:
                    achieved_result = "exceed"
            else:
                for temp_node in topo_graph.intention_nodes:
                    if(temp_node.intention_type==2):
                        temp_node.intention_type = 1
                        '''
                        temp_parent_node = temp_node.parent_node
                        temp_parent_node.sub_intentions.remove(temp_node)
                        topo_graph.intention_nodes.remove(temp_node)
                        topo_graph.all_nodes.remove(temp_node)
                        '''

                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=env_args.graph_train)
                # new_recheck_train
                if(HabitatAction.episode_train_step>=env_args.graph_episode_length-1 and action_node.node_type=="frontier_node" and achieved_result=="achieved") or (HabitatAction.episode_train_step>=env_args.graph_episode_length-1 and action_node.node_type=="intention_node" and action_node.intention_type!=2 and achieved_result=="achieved"):
                    achieved_result = "EXCEED_RL" # 超过RL最大次数 
                # new_recheck_train
                print("======> achieved_result <=====", achieved_result)
                print("=====> action_node_type <=====", action_node.node_type)
            
            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, graph_train=env_args.graph_train, rl_graph=rl_graph, policy=policy, topo_graph=topo_graph, scene_area=area_dict[habitat_env.current_episode.scene_id])
            
            if(action_node.node_type=="intention_node") and (action_node.intention_type==1) and (action_node in topo_graph.all_nodes):
                # if(topo_graph.current_node.name==action_node.parent_node.name):
                #     action_node_in_current_loc = np.array([action_node.rela_cx, action_node.rela_cy])
                # else:
                #     n_in_current_node = topo_graph.current_node.all_other_nodes_loc[action_node.parent_node.name]
                #     action_node_in_current_loc = get_absolute_pos(np.array([action_node.rela_cx, action_node.rela_cy]), n_in_current_node[:2], n_in_current_node[2])
                
                # now_action_dis = ((topo_graph.rela_cx-action_node_in_current_loc[0])**2+(topo_graph.rela_cy-action_node_in_current_loc[1])**2)**0.5
                
                world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                
                
                # if (now_action_dis<1):
                #     action_node.intention_type = 2
                #     # 距离1m以内的intention_node全部变为类型为2的intention_node
                #     for temp_node in topo_graph.intention_nodes:
                #         if(temp_node.parent_node.name==action_node.parent_node.name):
                #             temp_node_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                #         else:
                #             temp_parent_in_action_parent = action_node.parent_node.all_other_nodes_loc[temp_node.parent_node.name]
                #             temp_node_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_action_parent[:2], temp_parent_in_action_parent[2])
                        
                #         if (((temp_node_loc[0]-action_node.rela_cx)**2+(temp_node_loc[1]-action_node.rela_cy)**2)**0.5)<1:
                #             temp_node.intention_type = 2
            

                if (now_action_dis<1):
                    action_node.intention_type = 2
                    # 距离1m以内的intention_node全部变为类型为2的intention_node
                    for temp_node in topo_graph.intention_nodes:
                        if (((temp_node.world_cx-action_node.world_cx)**2+(temp_node.world_cy-action_node.world_cy)**2)**0.5)<1:
                            temp_node.intention_type = 2





            if(achieved_result=="exceed" or achieved_result=="empty" or achieved_result=="block" or achieved_result=="Failed_Plan" or evaluate_res=="false_reward" or achieved_result=="EXCEED_RL"):
                if(len(policy.buffer.reward_arrives)!=0):
                    policy.delete_buffer()

            if(achieved_result=="exceed" or achieved_result=="empty" or achieved_result=="block" or achieved_result=="Failed_Plan" or achieved_result=="EXCEED_RL"):
                break

            # =============> reward_revise <=============
            if(evaluate_res=="false_reward"):
                break
            # =============> reward_revise <=============
            
            # 剩余情况只能是achieved_result=="achieved" or achieved_result=="EXCEED_RL"
            reward = rl_graph.data['reward']
            done = rl_graph.data['arrive']
            reward_arrive = rl_graph.data['reward_arrive']

            reward_per_update += reward
            episode_total_reward += reward

            rl_graph.update(topo_graph)
            next_state = copy.deepcopy(rl_graph.data['state']) # 获得next_state

            policy.update_buffer(current_state, policy_acton_idx, next_state, reward, done, action_logprob, reward_arrive) # 一个样本

            current_state = copy.deepcopy(next_state) # 迭代更新

            policy.train_step += 1
            HabitatAction.episode_train_step += 1

            # =====> Train <=====
            # if(policy.train_step%rl_args.update_timestep)==0:
            if(len(policy.buffer.reward_arrives)%rl_args.update_timestep)==0:
                policy.delete_buffer()
                policy.update_policy(writer) # 重新计算reward+将网络参数更新k次+更新policy_old的参数
                policy.buffer.clear_memory() # 清空buffer

                model_pre_dir = '{0}/{1}/policy/{2}'.format(env_args.root, env_args.model_file_name, experiment_details)
                if not os.path.exists(model_pre_dir):
                    os.makedirs(model_pre_dir)
                    print(f"The new path:'{model_pre_dir}' has beed craeted!")

                policy.save('{0}/{1}/policy/{2}/{3}'.format(env_args.root, env_args.model_file_name, experiment_details, total_model_index))
                writer.add_scalar('Result/reward_per_update', reward_per_update/rl_args.update_timestep, total_model_index)

                reward_per_update = 0
                total_model_index += 1
                

            if(evaluate_res=="episode_stop"):
                # # ====> save_model <====
                # if ((Evaluate.real_episode_num_in_train-1) % env_args.graph_model_save_frequency == 0):
                #     policy.save('{0}/{1}/policy/{2}/{3}'.format(env_args.root, env_args.model_file_name, experiment_details, Evaluate.real_episode_num_in_train-1))
                # system_info(init_process_memory, init_free_memory)
                writer.add_scalar('Result/episode_total_reward', episode_total_reward, valid_episode_index)
                valid_episode_index += 1
                break
    