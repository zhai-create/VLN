import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import cv2
import habitat
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env_tools.arguments import args
from env_tools.data_utils import hm3d_config, init_gt_sensor, habitat_camera_intrinsic
from env_tools.evaluate_utils import Evaluate

from policy.rl_algorithms.arguments import args as rl_args
from system_utils import process_info
from policy.tools.utils import init_RL
from policy.rl_algorithms.rl_graph import RL_Graph


from perception.tools import fix_depth, get_rgb_image_ls, get_gt_image_ls
from graph.graph_utils import GraphMap
from graph.node_utils import Node

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

from perception.arguments import args as perception_args

from perception.intention_utils_rcnn import object_detect
from perception.intention_utils_gt import object_detect_gt

from vis_tools.vis_utils import init_mp4, get_top_down_map

if __name__=="__main__":
    args.task_stage = "val"
    args.graph_train = False
    args.root = "/home/zhaishichao/Data/VLN"
    if(args.is_llm==1 or args.is_llm==2):
        args.model_file_name = "Models_train_llm"
    else:
        args.model_file_name = "Models_train"

    if(args.is_llm==2):
        val_note = "_four_dim_small_thre_one_rgb_large_bs_train_val"
    elif(args.is_llm==1):
        val_note = "_three_dim_small_thre_one_rgb_large_bs_train_val"
    else:
        val_note = "_two_dim_one_depth_rotation_train_val_ji_reward_revise_12_factor_fake_intention_gt"
    
    if(args.is_llm==1 or args.is_llm==2):
        args.logger_file_name = "./log_files_llm/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    else:
        args.logger_file_name = "./log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.graph_episode_num = 1
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

    init_free_memory, init_process_memory = process_info()
    habitat_config = hm3d_config(stage=args.task_stage, episodes=args.graph_episode_num, max_steps=args.max_steps)

    for temp_pre_model in range(60, 80000, 10):
        args.graph_pre_model = temp_pre_model
        # experiment_details = 'graph_'  + rl_args.graph_task + '_' + rl_args.graph_action_space + \
        #     '_'+ rl_args.graph_encoder
        experiment_details = "graph_object_goal_navigation_adjacent_GAT_2025_03_04_16_18_28_two_dim_one_depth_rotation_reward_revise_12_factor_fake_inetntion_gt"
        
        while not os.path.exists("/home/zhaishichao/Data/VLN/{}/policy/{}/{}_critic".format(args.model_file_name, experiment_details, args.graph_pre_model)):
            print("not exists!!!")
            time.sleep(1)
        
        Evaluate.reset()
        
        habitat_env = habitat.Env(config=habitat_config)
        perception_args.intrinsic_matrix = habitat_camera_intrinsic(config=habitat_config)


        policy = init_RL(args, rl_args, experiment_details)

        # =====> select episodes <=====
        selected_episodes_scene_1 = []
        selected_episodes_scene_2 = []
        for index, temp_episode in enumerate(habitat_env.episodes):
            if(temp_episode.scene_id=="./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/val/00831-yr17PDCnDDW/yr17PDCnDDW.basis.glb"):
                selected_episodes_scene_1.append(temp_episode)
            elif(temp_episode.scene_id=="./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb"):
                if(temp_episode.episode_id=="0" or temp_episode.episode_id=="13"):
                    selected_episodes_scene_2.append(temp_episode)

        selected_episodes = selected_episodes_scene_1+selected_episodes_scene_2
        assert len(selected_episodes) == 30
        # =====> select episodes <=====

        for index_in_episodes in range(30):
            # rl_graph_init
            rl_graph = RL_Graph()
            # haitat_episode_init
            habitat_env.episodes = [selected_episodes[index_in_episodes]]
            observations = habitat_env.reset()

            
            object_goal = args.object_ls[observations["objectgoal"][0]]
            HabitatAction.reset(habitat_env, object_goal) 
            print("scene_id:", (habitat_env.current_episode.scene_id, object_goal))
            # print("=====> object_goal <=====", object_goal)
            # topo_graph_init
            topo_graph = GraphMap(habitat_env=habitat_env)
            topo_graph.set_current_pos(rela_cx=0.0, rela_cy=0.0, rela_turn=0.0)
            graph_update_flag = topo_graph.update()
            
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
                        if not habitat_env.episode_over:
                            habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                            observations = habitat_env.step(habitat_action)
                            achieved_result = "empty"
                        else:
                            achieved_result = "exceed"
                        Evaluate.evaluate(writer, achieved_result=achieved_result, habitat_env=habitat_env, action_node=None, index_in_episodes=index_in_episodes)
                        break

                action_node = rl_graph.all_nodes[polict_action]
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
                
                print("======> achieved_result <=====", achieved_result)
                print("=====> action_node_type <=====", action_node.node_type)
                
                evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, topo_graph=topo_graph)
                if(evaluate_res=="episode_stop"):
                    break

        writer.add_scalar('Val_Result/success_num', Evaluate.success_num, temp_pre_model)
        writer.add_scalar('Val_Result/spl_mean', Evaluate.spl_mean, temp_pre_model)
        writer.add_scalar('Val_Result/reward', (Evaluate.success_num*40+(-1)*Evaluate.all_front_steps/Evaluate.max_front_steps_per_rl_step)/30, temp_pre_model)
        # writer.add_scalar('Val_Result/reward', (Evaluate.success_num*40+(-1)*Evaluate.all_count_steps/Evaluate.max_count_steps_per_rl_step)/30, temp_pre_model)
        




