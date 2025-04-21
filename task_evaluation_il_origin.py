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
from policy.tools.utils import init_RL, init_IL
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
from perception.intention_utils_rcnn import object_detect
# from perception.intention_utils_dino import object_detect_sam
# from perception.intention_utils_gt import object_detect_gt
from perception.intention_utils_gt_other import object_detect_gt_other


if __name__=="__main__":
    args.task_stage = "val"
    args.graph_train = False
    args.root = "/home/zhaishichao/Data/VLN"
    if(args.is_llm==1 or args.is_llm==2):
        args.model_file_name = "Models_train_llm"
    else:
        args.model_file_name = "Models_train"
    args.graph_pre_model = 1169

    if(args.is_llm==2):
        val_note = "_four_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    elif(args.is_llm==1):
        val_note = "_three_dim_small_thre_one_rgb_large_bs_val_"+str(args.graph_pre_model)
    else:
        # val_note = "_multi_check_long_short_check_series_gt_val_"+str(args.graph_pre_model)
        # val_note = "_multi_check_il_semantic_relation_gt_val_"+str(args.graph_pre_model)
        val_note = "_multi_check_il_semantic_ls_relation_dis_revise_gt_val_"+str(args.graph_pre_model)+"_init_800"


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
        rl_args.graph_node_feature_dim = 24
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = True

    writer = SummaryWriter(args.logger_file_name)

    habitat_config = hm3d_config(stage=args.task_stage, episodes=args.graph_episode_num, max_steps=args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)
    perception_args.intrinsic_matrix = habitat_camera_intrinsic(config=habitat_config)

    init_free_memory, init_process_memory = process_info()
    policy = init_IL(args, rl_args)

    # false_index_ls = [4]
    # false_index_ls = [3, 6, 8, 11, 14, 15, 23, 24, 25, 29, 41, 50, 55, 58, 73, 76, 82, 92, 106, 115, 116, 127, 130, 138, 143, 148, 150, 154, 163, 180, 183, 184, 185, 192, 193, 195, 205, 216, 217, 218, 224, 230, 235, 238, 241, 243, 253, 255, 257, 265, 266, 267, 272, 273, 276, 278, 279, 299, 300, 301, 302, 305, 307, 310, 312, 318, 319, 322, 324, 328, 332, 360, 368, 379, 386, 390, 391, 394, 401, 402, 403, 409, 412, 415, 419, 421, 423, 426, 431, 433, 437, 438, 443, 446, 447, 450, 460, 462, 464, 470, 472, 474, 487, 490, 492, 496, 510, 518, 528, 533, 539, 552, 553, 556, 557, 574, 575, 584, 589, 601, 602, 603, 611, 626, 634, 640, 643, 652, 659, 664, 670, 671, 679, 680, 682, 684, 685, 691, 692, 693, 696, 701, 703, 708, 709, 715, 716, 728, 731, 736, 743, 744, 758, 763, 764, 775, 778, 782, 787, 788, 789, 790, 792, 793, 799, 814, 821, 822, 827, 829, 830, 833, 854, 856, 861, 862, 871, 872, 877, 880, 894, 910, 914, 915, 926, 928, 936, 942, 945, 946, 949, 962, 963, 974, 977, 983, 988, 995, 996, 997]

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

        if(index_in_episodes<800):
            continue

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

            # detect_res_pos_dict = object_detect_gt(gt_image_ls, depth, object_goal, HabitatAction.object_id_num_ls)
            # topo_graph.add_intention_gt(detect_res_pos_dict)

            detect_res_pos_dict = object_detect(rgb_image_ls, depth, object_goal)
            topo_graph.add_intention(detect_res_pos_dict)

            other_res_pos_dict = object_detect_gt_other(gt_image_ls, depth, HabitatAction.other_object_id_num_ls)
            topo_graph.add_other_intention(other_res_pos_dict)

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
                # action_node = policy.greedy_select_action_gt_near_goal(rl_graph, world_cx, world_cy, habitat_env.current_episode)
                print("=====> real_action_selection <=====")
            else:
                ghost_patch_res = topo_graph.ghost_patch(habitat_env, object_goal)
                rl_graph.update(topo_graph)
                if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0) and (ghost_patch_res=="ok"):
                    print("=====> ghost_patch <=====")
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train) # 2
                    # world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)
                    # action_node = policy.greedy_select_action_gt_near_goal(rl_graph, world_cx, world_cy, habitat_env.current_episode)
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
                if(args.is_vis==True):
                    achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=rl_graph, occu_writer=occu_writer, video_writer=video_writer, map_writer=map_writer, gt_writer=gt_writer)
                else:
                    achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
                print("======> achieved_result <=====", achieved_result)
                print("=====> action_node_type <=====", action_node.node_type)
            
            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, topo_graph=topo_graph)
            if(action_node.node_type=="intention_node") and (action_node.intention_type==1) and (action_node in topo_graph.all_nodes):
                world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env) # 当前机器人的位置
                now_action_dis = ((world_cx-action_node.world_cx)**2+(world_cy-action_node.world_cy)**2)**0.5
                if (now_action_dis<1):
                    action_node.intention_type = 2
                    # 距离1m以内的intention_node全部变为类型为2的intention_node
                    for temp_node in topo_graph.intention_nodes:
                        if (((temp_node.world_cx-action_node.world_cx)**2+(temp_node.world_cy-action_node.world_cy)**2)**0.5)<1:
                            temp_node.intention_type = 2
            
            
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