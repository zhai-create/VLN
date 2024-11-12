import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1, 2'
import cv2
import copy
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
from policy.tools.utils import init_RL, system_info
from policy.rl_algorithms.rl_graph import RL_Graph

from perception.tools import fix_depth, get_rgb_image_ls
from graph.graph_utils import GraphMap

from navigation.habitat_action import HabitatAction
from navigation.sub_goal_reach import SubgoalReach

if __name__=="__main__":
    args.task_stage = "train"
    args.graph_train = True
    args.root = "/home/zhaishichao/Data/VLN"
    args.model_file_name = "Models_train"
    args.graph_pre_model = 0

    train_note = "_two_dim_train" # 注释当前训练处于什么阶段
    date_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.logger_file_name = "./log_files_train/log_"+date_time+train_note
    args.graph_episode_num = 10000
    args.success_distance = 1.0
    args.max_steps = 10000

    # only_train
    args.graph_model_save_frequency = 1
    args.graph_batch_size = 64
    args.graph_episode_length = 40 # max train rl_steps for per episode
    

    rl_args.score_top_k = 50
    rl_args.graph_node_feature_dim = 2
    rl_args.graph_edge_feature_dim = 3
    rl_args.graph_embedding_dim = 64
    rl_args.graph_num_action_padding = 500
    rl_args.graph_num_graph_padding = -1
    rl_args.graph_sac_greedy = False

    # new_revise
    args.graph_iter_per_step = 1
    rl_args.lr_tune = 0.5e-3
    rl_args.graph_lr_actor = 0.5e-4
    rl_args.graph_lr_critic = 0.5e-4
    rl_args.random_exploration_length = 500

    # only_train
    rl_args.load_buffer_data_cnt = -1
    rl_args.save_buffer_data_path = "buffer_data/{}/".format(date_time)
    rl_args.load_buffer_data_path = ""

    writer = SummaryWriter(args.logger_file_name)

    habitat_config = hm3d_config(stage=args.task_stage, episodes=args.graph_episode_num, max_steps=args.max_steps)
    habitat_env = habitat.Env(config=habitat_config)

    

    experiment_details = 'graph_'  + rl_args.graph_task + '_' + rl_args.graph_action_space + \
        '_'+ rl_args.graph_encoder+"_"+date_time+train_note
    init_free_memory, init_process_memory = process_info()
    policy = init_RL(args, rl_args, experiment_details, writer=writer)

    for index_in_episodes in tqdm(range(args.graph_episode_num)):
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        observations = habitat_env.reset()

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

        # rl_graph_update
        rl_graph.update(topo_graph)
        current_state = copy.deepcopy(rl_graph.data['state'])

        # 用于手动调试
        if(args.is_auto==False):
            cv2.imshow("rgb", rgb_image_ls[0])
            cv2.imshow("depth", observations["depth"])
            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occu_for_show", occu_for_show)

        while True:    
            if(int(np.sum(rl_graph.data['state']['action_mask'].cpu().numpy()))>0):
                polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train)
                print("=====> real_action_selection <=====")
            else:
                topo_graph.ghost_patch()
                if(len(topo_graph.frontier_nodes)>0):
                    rl_graph.update(topo_graph)
                    current_state = copy.deepcopy(rl_graph.data['state']) # 1106最新修改
                    polict_action, policy_acton_idx = policy.select_action(rl_graph.data['state'], if_train=args.graph_train)
                    print("=====> ghost_patch <=====")
                else:
                    # action_space为空，结束当前episode
                    if not habitat_env.episode_over: # 没有超过1w步的最大步长
                        habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                        observations = habitat_env.step(habitat_action)
                        achieved_result = "empty"
                    else:
                        achieved_result = "exceed"
                    Evaluate.evaluate(writer, achieved_result=achieved_result, habitat_env=habitat_env, action_node=None, index_in_episodes=index_in_episodes, graph_train=args.graph_train, rl_graph=rl_graph, policy=policy)
                    break

            action_node = rl_graph.all_nodes[polict_action]
            achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=args.graph_train)
            if(policy.train_step==args.graph_episode_length-1 and action_node.node_type=="frontier_node"):
                achieved_result = "EXCEED_RL" # 超过RL最大次数 
            
            print("======> achieved_result <=====", achieved_result)
            print("=====> action_node_type <=====", action_node.node_type)

            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes, graph_train=args.graph_train, rl_graph=rl_graph, policy=policy)
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
            for train_index in range(args.graph_iter_per_step):     
                train_step = policy.train(writer, train_index, args.graph_batch_size) 

            if(evaluate_res=="episode_stop"):
                # ====> save_model <====
                if (Evaluate.real_episode_num_in_train-1) % args.graph_model_save_frequency == 0:
                    policy.save('{0}/{1}/policy/{2}/{3}'.format(args.root, args.model_file_name, experiment_details, Evaluate.real_episode_num_in_train-1))
                    
                print("ReplayBuffer Size : ", len(policy.buffer))
                system_info(init_process_memory, init_free_memory)
                break
    