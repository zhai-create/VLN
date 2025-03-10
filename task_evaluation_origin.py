import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
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
from graph.tools import find_node_path, get_absolute_pos
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
    args.graph_pre_model = 240

    if(args.is_llm==2):
        val_note = "_four_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    elif(args.is_llm==1):
        val_note = "_three_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    else:
        val_note = "_two_dim_12_factor_three_layer_frontier_cluster_gt_show_"+str(args.graph_pre_model)
    
    if(args.is_llm==1 or args.is_llm==2):
        args.logger_file_name = "./log_files_llm/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    else:
        args.logger_file_name = "./log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.graph_episode_num = 1000
    args.success_distance = 1.0 
    args.max_steps = 500

    args.is_vis = True # 录制视频

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
    experiment_details = "graph_object_goal_navigation_adjacent_GAT_2025_03_07_09_30_13_two_dim_12_factor_three_layer_frontier_cluster_gt"
    init_free_memory, init_process_memory = process_info()
    policy = init_RL(args, rl_args, experiment_details)

    # max_step_ls = [1, 16, 17, 19, 28, 30, 41, 42, 46, 50, 54, 56, 62, 82, 85, 89, 90, 94, 99, 103, 106, 109, 111, 122, 132, 133, 135, 137, 138, 139, 141, 145, 147, 158, 160, 161, 162, 165, 167, 173, 188, 189, 190, 195, 196, 197, 199, 200, 204, 205, 210, 211, 213, 217, 218, 221, 225, 234, 238, 240, 241, 243, 245, 255, 267, 271, 272, 275, 278, 282, 299, 300, 304, 305, 306, 309, 313, 316, 320, 321, 328, 329, 336, 337, 343, 344, 346, 357, 454, 458, 460, 464, 466, 469, 472, 473, 474, 475, 476, 481, 484, 487, 494, 526, 536, 541, 546, 557, 565, 575, 576, 578, 581, 583, 584, 587, 589, 603, 604, 609, 614, 616, 621, 628, 630, 639, 644, 647, 650, 672, 680, 683, 685, 697, 700, 705, 707, 711, 722, 724, 725, 729, 731, 732, 735, 739, 743, 744, 748, 756, 757, 760, 763, 764, 771, 772, 773, 779, 784, 785, 787, 788, 789, 790, 794, 796, 802, 806, 807, 809, 811, 812, 815, 819, 827, 832, 838, 839, 841, 842, 848, 851, 853, 855, 859, 871, 874, 892, 895, 904, 906, 914, 916, 921, 927, 933, 935, 943, 966, 976, 977, 980, 992, 994, 995, 997, 998]
    for index_in_episodes in tqdm(range(args.graph_episode_num)):   
        # 用于录制视频
        if(args.is_vis==True):
            occu_writer, video_writer, map_writer, gt_writer = init_mp4(pre_model=args.graph_pre_model, episode_index=index_in_episodes+1)
            get_top_down_map(habitat_env)
        
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        print("=====> scene_id <=====", habitat_env.current_episode.scene_id)

        observations = habitat_env.reset()
        object_goal = args.object_ls[observations["objectgoal"][0]]
        print("=====> object_goal <=====", object_goal)

        # if((index_in_episodes+1) not in max_step_ls):
        #     continue
        # if(index_in_episodes<800):
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
            topo_graph.current_node.update_occupancy(topo_graph.laser_2d_filtered, topo_graph.laser_2d_filtered_angle, np.array([topo_graph.rela_cx, topo_graph.rela_cy]), topo_graph.rela_turn)
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
                polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train)
                print("=====> real_action_selection <=====")
            else:
                ghost_patch_res = topo_graph.ghost_patch(habitat_env, object_goal)
                if(len(topo_graph.frontier_nodes)>0) and (ghost_patch_res=="ok"):
                    rl_graph.update(topo_graph)
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train)
                    print("=====> ghost_patch <=====")
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

            action_node = rl_graph.all_nodes[polict_action]
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