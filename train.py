import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import random
import cv2
import copy
import habitat
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from env_tools.arguments import args as env_args
from env_tools.data_utils import hm3d_config
from env_tools.evaluate_utils import Evaluate

from policy.rl_algorithms.arguments import args as rl_args
from system_utils import process_info
from policy.tools.utils import init_RL, system_info
from policy.rl_algorithms.rl_graph import RL_Graph

from perception.tools import fix_depth, get_rgb_image_ls
from perception.arguments import args as perception_args
from graph.graph_utils import GraphMap

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

if __name__=="__main__":
    env_args.task_stage = "train"
    env_args.graph_train = True
    env_args.root = "/home/zhaishichao/Data/VLN"
    env_args.model_file_name = "Models_train_llm"
    # env_args.model_file_name = "Models_train"
    env_args.graph_pre_model = 0

    # train_note = "_four_dim_new_question" # 注释当前训练处于什么阶段
    # train_note = "_two_dim_small_thre_one_rgb" # 注释当前训练处于什么阶段
    train_note = "_four_dim_baseline_llm_large_bs" # 注释当前训练处于什么阶段
    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    env_args.logger_file_name = "./log_files_train_llm/log_"+date_time+train_note
    # env_args.logger_file_name = "./log_files_train/log_"+date_time+train_note
    env_args.graph_episode_num = 80000
    env_args.success_distance = 1.0
    env_args.max_steps = 10000

    # only_train
    env_args.graph_model_save_frequency = 1
    env_args.graph_batch_size = 128
    # env_args.graph_batch_size = 64
    env_args.graph_episode_length = 40 # max train rl_steps for per episode
    

    rl_args.score_top_k = 50
    rl_args.graph_node_feature_dim = 4
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
    rl_args.random_exploration_length = 400
    random.seed(456)

    # only_train
    rl_args.load_buffer_data_cnt = -1
    rl_args.save_buffer_data_path = "buffer_data/{}/".format(date_time)
    rl_args.load_buffer_data_path = ""

    writer = SummaryWriter(env_args.logger_file_name)

    habitat_config = hm3d_config(stage=env_args.task_stage, episodes=1, max_steps=env_args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)

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
        # writer.add_scalar('Scene/scene_id', list(id_dict.keys()).index(habitat_env.episodes[0].scene_id), index_in_episodes+1)
        print("=====> scene_id <=====", habitat_env.episodes[0].scene_id)
        try:
            observations = habitat_env.reset()
        except:
            continue
        HabitatAction.reset(habitat_env) 

        habitat_metric = habitat_env.get_metrics()
        object_goal = env_args.object_ls[observations["objectgoal"][0]]

        print("=====> object_goal <=====", object_goal)
        # get sensor data: rgb, depth, 2d_laser
        rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
        depth = fix_depth(observations["depth"])

        # topo_graph_init
        topo_graph = GraphMap(habitat_env=habitat_env)
        topo_graph.set_current_pos(rela_cx=0.0, rela_cy=0.0, rela_turn=0.0)
        topo_graph.update(rgb_image_ls, depth, object_goal)

        # rl_graph_update
        rl_graph.update(topo_graph)
        current_state = copy.deepcopy(rl_graph.data['state'])

        # 用于手动调试
        if(env_args.is_auto==False):
            cv2.imshow("rgb", rgb_image_ls[0])
            cv2.imshow("depth", observations["depth"])
            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occu_for_show", occu_for_show)

        while True:    
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=env_args.graph_train)
                print("=====> real_action_selection <=====")
            else:
                topo_graph.ghost_patch()
                if(len(topo_graph.frontier_nodes)>0):
                    rl_graph.update(topo_graph)
                    current_state = copy.deepcopy(rl_graph.data['state']) # 1106最新修改
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=env_args.graph_train)
                    print("=====> ghost_patch <=====")
                else:
                    # action_space为空，结束当前episode
                    if not habitat_env.episode_over: # 没有超过1w步的最大步长
                        habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                        observations = habitat_env.step(habitat_action)
                        achieved_result = "empty"
                    else:
                        achieved_result = "exceed"
                    Evaluate.evaluate(writer, achieved_result=achieved_result, habitat_env=habitat_env, action_node=None, index_in_episodes=index_in_episodes, graph_train=env_args.graph_train, rl_graph=rl_graph, policy=policy)
                    break

            action_node = rl_graph.all_nodes[polict_action]
            achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=env_args.graph_train)
            if(policy.train_step==env_args.graph_episode_length-1 and action_node.node_type=="frontier_node"):
                achieved_result = "EXCEED_RL" # 超过RL最大次数 
            
            print("======> achieved_result <=====", achieved_result)
            print("=====> action_node_type <=====", action_node.node_type)

            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, graph_train=env_args.graph_train, rl_graph=rl_graph, policy=policy)
            if(achieved_result=="block" or achieved_result=="Failed_Plan" or achieved_result=="exceed"):
                break
            # 剩余情况只能是achieved_result=="achieved" or achieved_result=="EXCEED_RL"
            rl_graph.update(topo_graph)
            next_state = copy.deepcopy(rl_graph.data['state'])
            reward = rl_graph.data['reward']
            done = rl_graph.data['arrive']
            policy.update_buffer(current_state, policy_acton_idx, next_state, reward, done, 0) # 一个样本
            current_state = copy.deepcopy(next_state) # 迭代更新

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
    