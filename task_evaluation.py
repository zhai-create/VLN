import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import cv2
import habitat
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env_tools.arguments import args
from env_tools.data_utils import hm3d_config
from env_tools.evaluate_utils import Evaluate

from policy.rl_algorithms.arguments import args as rl_args
from system_utils import process_info
from policy.tools.utils import init_RL
from policy.rl_algorithms.rl_graph import RL_Graph


from perception.tools import fix_depth, get_rgb_image_ls
from graph.graph_utils import GraphMap

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

from vis_tools.vis_utils import init_mp4, get_top_down_map

from perception.arguments import args as perception_args

if __name__=="__main__":
    args.task_stage = "val"
    args.graph_train = False
    args.root = "/home/zhaishichao/Data/VLN"
    # args.model_file_name = "Models_train_llm"
    args.model_file_name = "Models_train"
    args.graph_pre_model = 1735

    # val_note = "_four_dim_"+str(args.graph_pre_model) # 注释当前测试处于什么阶段
    val_note = "_two_dim_small_thre_one_rgb_"+str(args.graph_pre_model) # 注释当前测试处于什么阶段
    # val_note = "_four_dim_baseline_llm_large_bs_val_"+str(args.graph_pre_model) # 注释当前测试处于什么阶段
    # args.logger_file_name = "./log_files_llm/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.logger_file_name = "./log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.graph_episode_num = 1000
    args.success_distance = 1.0 
    args.max_steps = 500

    args.is_vis = False # 录制视频

    rl_args.score_top_k = 50
    rl_args.graph_node_feature_dim = 2
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = True

    writer = SummaryWriter(args.logger_file_name)

    habitat_config = hm3d_config(stage=args.task_stage, episodes=args.graph_episode_num, max_steps=args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)

    # experiment_details = 'graph_'  + rl_args.graph_task + '_' + rl_args.graph_action_space + \
    #     '_'+ rl_args.graph_encoder
    experiment_details = "graph_object_goal_navigation_adjacent_GAT_2024_12_16_20_27_24_two_dim_small_thre_one_rgb"
    # experiment_details = "graph_object_goal_navigation_adjacent_GAT_2024_12_17_18_25_15_four_dim_baseline_llm_large_bs"

    init_free_memory, init_process_memory = process_info()
    policy = init_RL(args, rl_args, experiment_details)

    for index_in_episodes in tqdm(range(args.graph_episode_num)):   
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        observations = habitat_env.reset()
        # if(index_in_episodes<800):
        #     continue

        HabitatAction.reset(habitat_env) 
        habitat_metric = habitat_env.get_metrics()
        object_goal = args.object_ls[observations["objectgoal"][0]]

        print("=====> object_goal <=====", object_goal)
        # get sensor data: rgb, depth, 2d_laser
        rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
        depth = fix_depth(observations["depth"])
        
        # topo_graph_init
        topo_graph = GraphMap(habitat_env=habitat_env)
        topo_graph.set_current_pos(rela_cx=0.0, rela_cy=0.0, rela_turn=0.0)
        topo_graph.update(rgb_image_ls, depth, object_goal)

        # 用于手动调试
        if(args.is_auto==False):
            cv2.imshow("rgb", rgb_image_ls[0])
            cv2.imshow("depth", observations["depth"])
            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occu_for_show", occu_for_show)

        # 用于录制视频
        if(args.is_vis==True):
            video_writer, map_writer = init_mp4(pre_model=args.graph_pre_model, episode_index=index_in_episodes+1)
            get_top_down_map(habitat_env)

        while True:
            # rl_graph_update
            rl_graph.update(topo_graph)
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train)
                print("=====> real_action_selection <=====")
            else:
                topo_graph.ghost_patch()
                if(len(topo_graph.frontier_nodes)>0):
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
            # achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, index_in_episodes=index_in_episodes, writer=writer)
            
            if(args.is_vis==True):
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=rl_graph, video_writer=video_writer, map_writer=map_writer)
            else:
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
            print("======> achieved_result <=====", achieved_result)
            print("=====> action_node_type <=====", action_node.node_type)
            
            rl_graph_action_node_name_ls = [(temp_node.name, temp_node.score) for temp_node in rl_graph.all_nodes if(temp_node.node_type!="explored_node")]
            print("=====> rl_graph_action_node_name_ls <=====", rl_graph_action_node_name_ls)
            print("============> intention_score <=============", action_node.score)
            
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




