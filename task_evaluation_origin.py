import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import cv2
import habitat
import habitat_sim
import datetime
import random

import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env_tools.arguments import args
from env_tools.data_utils import hm3d_config, habitat_camera_intrinsic
from env_tools.evaluate_utils import Evaluate

from policy.rl_algorithms.arguments import args as rl_args
from system_utils import process_info
from policy.tools.utils import init_RL
from policy.rl_algorithms.rl_graph import RL_Graph


from perception.tools import fix_depth, get_rgb_image_ls, get_gt_image_ls
from graph.graph_utils import GraphMap
from graph.tools import find_node_path, get_absolute_pos, get_current_world_pos
from graph.arguments import args as graph_args
from graph.node_utils import Node

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

from vis_tools.vis_utils import init_mp4, get_top_down_map, save_mp4

from perception.arguments import args as perception_args
# from perception.intention_utils_rcnn import object_detect
# from perception.intention_utils_dino import object_detect_sam
from perception.intention_utils_gt import object_detect_gt


if __name__=="__main__":
    args.task_stage = "val"
    args.graph_train = False
    args.root = "/home/zhaishichao/Data/VLN"
    if(args.is_llm==1 or args.is_llm==2):
        args.model_file_name = "Models_train_llm"
    else:
        args.model_file_name = "Models_train"
    args.graph_pre_model = 150

    if(args.is_llm==2):
        val_note = "_four_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    elif(args.is_llm==1):
        val_note = "_three_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    else:
        # val_note = "_multi_check_long_short_check_series_gt_val_"+str(args.graph_pre_model)
        val_note = "_single_check_fake_intention_gt_val_"+str(args.graph_pre_model)
        # val_note = "_single_check_fake_intention_greedy_gt_val"
        # val_note = "_single_check_greedy_gt_val"
    
    if(args.is_llm==1 or args.is_llm==2):
        args.logger_file_name = "./log_files_llm/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    else:
        args.logger_file_name = "./log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.graph_episode_num = 1000
    args.success_distance = 1.0 
    args.max_steps = 500

    args.is_vis = False # 录制视频

    rl_args.score_top_k = 50
    if(args.is_llm==2):
        rl_args.graph_node_feature_dim = 4
    elif(args.is_llm==1):
        rl_args.graph_node_feature_dim = 3
    else:
        rl_args.graph_node_feature_dim = 2
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = True

    writer = SummaryWriter(args.logger_file_name)

    habitat_config = hm3d_config(stage=args.task_stage, episodes=args.graph_episode_num, max_steps=args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)
    perception_args.intrinsic_matrix = habitat_camera_intrinsic(config=habitat_config)

    # experiment_details = 'graph_'  + rl_args.graph_task + '_' + rl_args.graph_action_space + \
    #     '_'+ rl_args.graph_encoder
    # experiment_details = "graph_object_goal_navigation_adjacent_GAT_2025_01_10_05_23_47_two_dim_small_thre_rgb_new_framework"
    # experiment_details = "graph_object_goal_navigation_adjacent_GAT_2025_01_14_10_27_04_two_dim_small_thre_cluster_recheck"
    experiment_details = "graph_object_goal_navigation_adjacent_GAT_2025_04_05_14_53_16_single_check_fake_intention_gt_train"
    init_free_memory, init_process_memory = process_info()
    policy = init_RL(args, rl_args, experiment_details)

    # false_index_ls = [1, 17, 18, 19, 28, 41, 42, 50, 51, 53, 54, 62, 67, 80, 81, 89, 90, 94, 99]
    # false_index_ls = [23, 24, 26, 27, 29, 34, 41, 42, 46, 47, 49, 50]
    # false_index_ls = [42, 49, 64]

    for index_in_episodes in tqdm(range(args.graph_episode_num)):   
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        print("=====> scene_id <=====", habitat_env.current_episode.scene_id)

        observations = habitat_env.reset()

        # 用于录制视频
        if(args.is_vis==True):
            occu_writer, video_writer, map_writer, gt_writer = init_mp4(pre_model=args.graph_pre_model, episode_index=index_in_episodes+1)
            get_top_down_map(habitat_env, observations)


        object_goal = args.object_ls[observations["objectgoal"][0]]
        print("=====> object_goal <=====", object_goal)

        # if((index_in_episodes+1) not in false_index_ls):
        #     continue

        # if(index_in_episodes<3):
        #     continue

        HabitatAction.reset(habitat_env, object_goal, args.graph_train) 
        habitat_metric = habitat_env.get_metrics()
        
        # topo_graph_init
        topo_graph = GraphMap(habitat_env=habitat_env)
        topo_graph.set_current_pos(rela_cx=0.0, rela_cy=0.0, rela_turn=0.0)
        graph_update_flag = topo_graph.update()
        
        # 用于录制视频
        if(args.is_vis==True):
            save_mp4(occu_writer, video_writer, map_writer, gt_writer, habitat_env, topo_graph, rl_graph, action_node=None, object_goal=object_goal)
        
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
            topo_graph.add_intention(detect_res_pos_dict, rgb_image_ls, object_goal)

            # 底层仿真器动作执行
            habitat_action = HabitatAction.set_habitat_action("r", topo_graph)
            observations = habitat_env.step(habitat_action)
            topo_graph.obs = observations


            # 用于录制视频
            if(args.is_vis==True):
                save_mp4(occu_writer, video_writer, map_writer, gt_writer, habitat_env, topo_graph, rl_graph, action_node=None, object_goal=object_goal)

        while True:
            # rl_graph_update
            rl_graph.update(topo_graph)
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train) # 1
                # world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
                # action_node = policy.gt_greedy_select_action(rl_graph, world_cx, world_cy)
                print("=====> real_action_selection <=====")
            else:
                ghost_patch_res = topo_graph.ghost_patch(habitat_env, object_goal)
                rl_graph.update(topo_graph)
                if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0) and (ghost_patch_res=="ok"):
                    print("=====> ghost_patch <=====")
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train) # 2
                    # world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
                    # action_node = policy.gt_greedy_select_action(rl_graph, world_cx, world_cy)
                else:
                    # action_space为空，结束当前episode
                    print("========> empty_action_space <========")
                    if not habitat_env.episode_over:
                        habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                        observations = habitat_env.step(habitat_action)
                        achieved_result = "empty"
                    else:
                        achieved_result = "exceed"
                    Evaluate.evaluate(writer, achieved_result=achieved_result, habitat_env=habitat_env, action_node=None, index_in_episodes=index_in_episodes)
                    break

            # 用于录制视频
            if(args.is_vis==True):
                save_mp4(occu_writer, video_writer, map_writer, gt_writer, habitat_env, topo_graph, rl_graph, action_node=None, object_goal=object_goal)

            action_node = rl_graph.all_nodes[polict_action] # 3
            if(args.is_vis==True):
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=rl_graph, occu_writer=occu_writer, video_writer=video_writer, map_writer=map_writer, gt_writer=gt_writer)
            else:
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
        
            print("======> achieved_result <=====", achieved_result)
            print("=====> action_node_type <=====", action_node.node_type)
            
            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, topo_graph=topo_graph)
            
            if(evaluate_res=="episode_stop"):
                # =====> new_add_evaluate <=====
                writer.add_scalar('Policy/selected_intention_score', action_node.score, index_in_episodes+1)
                all_intention_score_ls = [temp_node.score for temp_node in rl_graph.all_nodes if(temp_node.node_type=="intention_node")]
                all_frontier_score_ls = [temp_node.score for temp_node in rl_graph.all_nodes if(temp_node.node_type=="frontier_node")]
                writer.add_scalar('Policy/len_intention_nodes', len(all_intention_score_ls), index_in_episodes+1)
                writer.add_scalar('Policy/len_frontier_nodes', len(all_frontier_score_ls), index_in_episodes+1)
                if(len(all_intention_score_ls)>0):
                    writer.add_scalar('Policy/max_intention_score', max(all_intention_score_ls), index_in_episodes+1)
                    writer.add_scalar('Policy/min_intention_score', min(all_intention_score_ls), index_in_episodes+1)
                # =====> new_add_evaluate <=====
                break