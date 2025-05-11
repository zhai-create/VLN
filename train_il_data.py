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
    if(env_args.is_llm==1 or env_args.is_llm==2):
        env_args.model_file_name = "Models_train_llm"
    else:
        env_args.model_file_name = "Models_train"
    env_args.graph_pre_model = 0

    if(env_args.is_llm==2):
        train_note = "_four_dim_small_thre_one_rgb_large_bs" # 注释当前训练处于什么阶段
    elif(env_args.is_llm==1):
        train_note = "_three_dim_small_thre_one_rgb_large_bs" # 注释当前训练处于什么阶段
    else:
        # train_note = "_multi_check_large_punish_gt_train" # 注释当前训练处于什么阶段
        # train_note = "_multi_check_il_data_gt_0421_semantic_ls_near_goal" # 注释当前训练处于什么阶段
        train_note = "_multi_check_il_data_frontier_score_revise_intention_for_ring" # 注释当前训练处于什么阶段

    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if(env_args.is_llm==1 or env_args.is_llm==2):
        env_args.logger_file_name = "./log_files_train_llm/log_"+date_time+train_note
    else:
        env_args.logger_file_name = "./log_files_train/log_"+date_time+train_note
    env_args.graph_episode_num = 80000
    env_args.success_distance = 1.0
    env_args.max_steps = 10000

    # only_train
    env_args.graph_model_save_frequency = 1
    # env_args.graph_batch_size = 128
    env_args.graph_batch_size = 64
    env_args.graph_episode_length = 40 # max train rl_steps for per episode
    

    rl_args.score_top_k = 50
    if(env_args.is_llm==2):
        rl_args.graph_node_feature_dim = 4
    elif(env_args.is_llm==1):
        rl_args.graph_node_feature_dim = 3
    else:
        rl_args.graph_node_feature_dim = 152
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = False

    # new_revise
    env_args.graph_iter_per_step = 2
    rl_args.lr_tune = 0.5e-3
    # rl_args.graph_lr_actor = 0.5e-4
    # rl_args.graph_lr_critic = 0.5e-4
    rl_args.random_exploration_length = 8000000000
    
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


    # # =====> all scene episodes <=====
    # selected_episodes = copy.deepcopy(habitat_env.episodes)
    # random.shuffle(selected_episodes)
    # selected_episodes = selected_episodes[:args.graph_episode_num]
    # # =====> all scene episodes  <=====


    # # =====> all scene episodes <=====
    # selected_episodes = []
    # selected_scene_ls = []
    # for index, temp_episode in enumerate(habitat_env.episodes):
    #     if(temp_episode.scene_id not in selected_scene_ls):
    #         selected_scene_ls.append(temp_episode.scene_id)
    #         selected_episodes.append(temp_episode)
    #     if(len(selected_scene_ls)==79):
    #         break
    # # =====> all scene episodes  <=====

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
        HabitatAction.rotate_loc_ls.append([0, 0])
        
        # rl_graph_update
        rl_graph.update(topo_graph)
        current_state = copy.deepcopy(rl_graph.data['state'])
        world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
        current_min_goal_loc = get_min_goal_loc(habitat_env.current_episode, world_cx, world_cy)
        current_robot_loc = np.array([world_cx, world_cy])
        current_robot_rela_loc_in_init = np.array([0.0, 0.0])
        current_robot_rela_loc_in_current = np.array([topo_graph.rela_cx, topo_graph.rela_cy])
        current_node_name_for_calculate = topo_graph.current_node.name


        # 获得初始all_map_loc和初始intention_node的个数
        HabitatAction.get_all_map_loc(topo_graph)
        HabitatAction.get_all_see_intention(topo_graph, rl_graph)

        # 用于手动调试
        if(env_args.is_auto==False):
            cv2.imshow("rgb", rgb_image_ls[0])
            cv2.imshow("depth", observations["depth"])
            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occu_for_show", occu_for_show)

        while True:   
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                if(policy.train_step < 6000000000):
                    action_node = policy.greedy_select_action_ring_vlm_score(rl_graph, topo_graph)
                    polict_action = action_node.rl_node_index
                    policy_acton_idx = action_node.action_in_space_index
                else:
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=env_args.graph_train)
                    action_node = rl_graph.all_nodes[polict_action]
                print("=====> real_action_selection <=====")
            else:
                ghost_patch_res = topo_graph.ghost_patch(habitat_env, object_goal)
                rl_graph.update(topo_graph)
                if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0) and (ghost_patch_res=="ok"):
                    print("=====> ghost_patch <=====")
                    current_state = copy.deepcopy(rl_graph.data['state']) # 1106最新修改
                    world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
                    current_min_goal_loc = get_min_goal_loc(habitat_env.current_episode, world_cx, world_cy)
                    current_robot_loc = np.array([world_cx, world_cy])
                    current_node_name_for_calculate = topo_graph.current_node.name

                    if(topo_graph.current_node.name==topo_graph.explored_nodes[0].name):
                        current_robot_rela_loc_in_init = np.array([topo_graph.rela_cx, topo_graph.rela_cy])
                    else:
                        current_node_in_init_loc = topo_graph.explored_nodes[0].all_other_nodes_loc[topo_graph.current_node.name]
                        current_robot_rela_loc_in_init = get_absolute_pos(np.array([topo_graph.rela_cx, topo_graph.rela_cy]), current_node_in_init_loc[:2], current_node_in_init_loc[2])

                    current_robot_rela_loc_in_current = np.array([topo_graph.rela_cx, topo_graph.rela_cy])

                    if(policy.train_step < 6000000000):
                        action_node = policy.greedy_select_action_ring_vlm_score(rl_graph, topo_graph)
                        polict_action = action_node.rl_node_index
                        policy_acton_idx = action_node.action_in_space_index
                    else:
                        polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=env_args.graph_train)
                        action_node = rl_graph.all_nodes[polict_action]
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
                if(topo_graph.current_node.name==action_node.parent_node.name):
                    action_node_in_current_loc = np.array([action_node.rela_cx, action_node.rela_cy])
                else:
                    n_in_current_node = topo_graph.current_node.all_other_nodes_loc[action_node.parent_node.name]
                    action_node_in_current_loc = get_absolute_pos(np.array([action_node.rela_cx, action_node.rela_cy]), n_in_current_node[:2], n_in_current_node[2])
                
                now_action_dis = ((topo_graph.rela_cx-action_node_in_current_loc[0])**2+(topo_graph.rela_cy-action_node_in_current_loc[1])**2)**0.5
                if (now_action_dis<1):
                    action_node.intention_type = 2
                    # 距离1m以内的intention_node全部变为类型为2的intention_node
                    for temp_node in topo_graph.intention_nodes:
                        if(temp_node.parent_node.name==action_node.parent_node.name):
                            temp_node_loc = np.array([temp_node.rela_cx, temp_node.rela_cy])
                        else:
                            temp_parent_in_action_parent = action_node.parent_node.all_other_nodes_loc[temp_node.parent_node.name]
                            temp_node_loc = get_absolute_pos(np.array([temp_node.rela_cx, temp_node.rela_cy]), temp_parent_in_action_parent[:2], temp_parent_in_action_parent[2])
                        
                        if (((temp_node_loc[0]-action_node.rela_cx)**2+(temp_node_loc[1]-action_node.rela_cy)**2)**0.5)<1:
                            temp_node.intention_type = 2
            
            
            if(achieved_result=="block" or achieved_result=="Failed_Plan" or achieved_result=="exceed"):
                break

            # =============> reward_revise <=============
            if(evaluate_res=="false_reward"):
                break
            # =============> reward_revise <=============

            
            # 剩余情况只能是achieved_result=="achieved" or achieved_result=="EXCEED_RL"
            reward = rl_graph.data['reward']
            done = rl_graph.data['arrive']

            # 保存buffer数据
            il_data = {}
            il_data["current_state"] = current_state
            il_data["policy_acton_idx"] = policy_acton_idx
            il_data["current_min_goal_loc"] = current_min_goal_loc
            il_data["index_in_episodes"] = index_in_episodes # 从0开始
            il_data["current_robot_loc"] = current_robot_loc # 从0开始
            il_data["current_robot_rela_loc_in_init"] = current_robot_rela_loc_in_init
            il_data["current_robot_rela_loc_in_current"] = current_robot_rela_loc_in_current

            il_data["all_goal_loc_ls"] = [[temp_position.position[0], temp_position.position[1], temp_position.position[2]] for temp_position in habitat_env.current_episode.goals]
            il_data["polict_action"] = polict_action


            frontier_dict = {}
            for temp_frontier in rl_graph.all_frontier_nodes:
                if(temp_frontier.parent_node.name==topo_graph.explored_nodes[0].name):
                    temp_frontier_init_loc = np.array([temp_frontier.rela_cx, temp_frontier.rela_cy])
                else:
                    temp_frontier_parent_in_init = topo_graph.explored_nodes[0].all_other_nodes_loc[temp_frontier.parent_node.name]
                    temp_frontier_init_loc = get_absolute_pos(np.array([temp_frontier.rela_cx, temp_frontier.rela_cy]), temp_frontier_parent_in_init[:2], temp_frontier_parent_in_init[2])

                current_node_for_cal = topo_graph.get_node(current_node_name_for_calculate)

                if(temp_frontier.parent_node.name==current_node_for_cal.name):
                    temp_frontier_current_loc = np.array([temp_frontier.rela_cx, temp_frontier.rela_cy])
                else:
                    temp_frontier_parent_in_current = current_node_for_cal.all_other_nodes_loc[temp_frontier.parent_node.name]
                    temp_frontier_current_loc = get_absolute_pos(np.array([temp_frontier.rela_cx, temp_frontier.rela_cy]), temp_frontier_parent_in_current[:2], temp_frontier_parent_in_current[2])

                frontier_dict[temp_frontier.name] = [temp_frontier.rgb_image_ls, HabitatAction.object_goal, temp_frontier.rl_node_index , temp_frontier.action_in_space_index, temp_frontier.world_cx, temp_frontier.world_cy, temp_frontier_init_loc[0], temp_frontier_init_loc[1], temp_frontier_current_loc[0], temp_frontier_current_loc[1]]
            il_data["frontier_dict"] = frontier_dict

            intention_dict = {}
            for temp_intention in rl_graph.all_intention_nodes:
                if(temp_intention.parent_node.name==topo_graph.explored_nodes[0].name):
                    temp_intention_init_loc = np.array([temp_intention.rela_cx, temp_intention.rela_cy])
                else:
                    temp_intention_parent_in_init = topo_graph.explored_nodes[0].all_other_nodes_loc[temp_intention.parent_node.name]
                    temp_intention_init_loc = get_absolute_pos(np.array([temp_intention.rela_cx, temp_intention.rela_cy]), temp_intention_parent_in_init[:2], temp_intention_parent_in_init[2])

                current_node_for_cal = topo_graph.get_node(current_node_name_for_calculate)

                if(temp_intention.parent_node.name==current_node_for_cal.name):
                    temp_intention_current_loc = np.array([temp_intention.rela_cx, temp_intention.rela_cy])
                else:
                    temp_intention_parent_in_current = current_node_for_cal.all_other_nodes_loc[temp_intention.parent_node.name]
                    temp_intention_current_loc = get_absolute_pos(np.array([temp_intention.rela_cx, temp_intention.rela_cy]), temp_intention_parent_in_current[:2], temp_intention_parent_in_current[2])

                intention_dict[temp_intention.name] = [temp_intention.score, temp_intention.near_score_ls, temp_intention.init_dis, temp_intention.near_dis_ls, temp_intention.res_col_index_factor, temp_intention.near_res_col_index_factor_ls, HabitatAction.object_goal, temp_intention.intention_type, temp_intention.rl_node_index , temp_intention.action_in_space_index, temp_intention.world_cx, temp_intention.world_cy, temp_intention_init_loc[0], temp_intention_init_loc[1], temp_intention_current_loc[0], temp_intention_current_loc[1]]
            il_data["intention_dict"] = intention_dict

            rl_graph.update(topo_graph)
            next_state = copy.deepcopy(rl_graph.data['state'])
            policy.update_buffer(current_state, policy_acton_idx, next_state, reward, done, 0) # 一个样本
            
            # np.save('il_data/{}.npy'.format(policy.train_step+1+7735), il_data)
            save_root = "il_data_frontier_score_revise_intention_for_ring"
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            np.save('{}/{}.npy'.format(save_root, policy.train_step+1), il_data)
            

            current_state = copy.deepcopy(next_state) # 迭代更新
            world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
            current_min_goal_loc = get_min_goal_loc(habitat_env.current_episode, world_cx, world_cy)
            current_robot_loc = np.array([world_cx, world_cy])
            current_node_name_for_calculate = topo_graph.current_node.name

            if(topo_graph.current_node.name==topo_graph.explored_nodes[0].name):
                current_robot_rela_loc_in_init = np.array([topo_graph.rela_cx, topo_graph.rela_cy])
            else:
                current_node_in_init_loc = topo_graph.explored_nodes[0].all_other_nodes_loc[topo_graph.current_node.name]
                current_robot_rela_loc_in_init = get_absolute_pos(np.array([topo_graph.rela_cx, topo_graph.rela_cy]), current_node_in_init_loc[:2], current_node_in_init_loc[2])

            current_robot_rela_loc_in_current = np.array([topo_graph.rela_cx, topo_graph.rela_cy])

            # =====> Train <=====
            for train_index in range(env_args.graph_iter_per_step):     
                train_step = policy.train(writer, train_index, env_args.graph_batch_size) 

            if(evaluate_res=="episode_stop"):
                # ====> save_model <====
                if ((Evaluate.real_episode_num_in_train-1) % env_args.graph_model_save_frequency == 0):
                    policy.save('{0}/{1}/policy/{2}/{3}'.format(env_args.root, env_args.model_file_name, experiment_details, Evaluate.real_episode_num_in_train-1))
                    
                print("ReplayBuffer Size : ", len(policy.buffer))
                system_info(init_process_memory, init_free_memory)
                break
    