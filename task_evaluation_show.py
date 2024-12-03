import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1'
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
    args.is_vis = False

    args.task_stage = "val"
    args.graph_train = False
    args.root = "/home/zhaishichao/Data/VLN"
    args.model_file_name = "Models_train"
    # args.model_file_name = "Models"
    args.graph_pre_model = 399
    # args.graph_pre_model = 1
    

    val_note = "_compare_hm3d_"+str(args.graph_pre_model) # 注释当前测试处于什么阶段
    args.logger_file_name = "./log_files/log_"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+val_note
    args.graph_episode_num = 1000
    args.success_distance = 1.0 
    args.max_steps = 500

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
    # experiment_details = "graph_object_goal_navigation_adjacent_GAT"
    experiment_details = "graph_object_goal_navigation_adjacent_GAT_2024_11_28_02_07_13_two_dim_train_new_load_data_all_label_low_cost_15_scene"
    init_free_memory, init_process_memory = process_info()
    policy = init_RL(args, rl_args, experiment_details)

    # gibson_better_ls = [2, 3, 5, 16, 17, 19, 21, 22, 25, 26, 28, 30, 31, 32, 33, 35, 36, 37, 38, 43, 46, 49, 51, 52, 54, 55, 56, 57, 59, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 75, 77, 78, 80, 81, 82, 83, 85, 86, 88, 90, 91, 92, 94, 96, 104, 109, 110, 111, 112, 113, 114, 116, 118, 120, 122, 123, 125, 130, 133, 135, 141, 145, 147, 148, 149, 151, 153, 154, 159, 161, 162, 165, 167, 168, 175, 176, 177, 178, 179, 180, 181, 186, 190, 193, 194, 197, 201, 209, 211, 219, 221, 222, 225, 226, 229, 232, 233, 234, 237, 239, 240, 243, 246, 247, 249, 258, 268, 271, 272, 274, 280, 282, 284, 285, 287, 289, 292, 295, 298, 306, 307, 311, 313, 317, 321, 322, 325, 326, 327, 329, 331, 332, 336, 338, 342, 346, 347, 349, 350, 352, 353, 354, 356, 358, 360, 364, 365, 368, 369, 371, 376, 379, 383, 384, 385, 389, 390, 392, 393, 395, 397, 398, 402, 405, 410, 413, 414, 416, 418, 422, 424, 426, 432, 433, 434, 437, 439, 441, 447, 448, 451, 452, 454, 456, 458, 462, 464, 467, 475, 478, 479, 481, 486, 488, 489, 491, 492, 493, 494, 495, 496, 498, 499, 501, 508, 509, 510, 511, 512, 518, 519, 520, 523, 524, 528, 531, 532, 533, 536, 538, 540, 541, 544, 547, 552, 558, 560, 563, 566, 569, 578, 584, 586, 590, 592, 598, 607, 609, 610, 616, 624, 630, 635, 637, 639, 640, 642, 643, 645, 647, 651, 653, 654, 658, 659, 661, 662, 663, 665, 668, 669, 672, 674, 676, 677, 681, 683, 687, 688, 689, 690, 703, 704, 705, 711, 712, 713, 715, 720, 723, 726, 727, 730, 733, 735, 737, 741, 746, 748, 756, 757, 759, 760, 764, 765, 766, 768, 771, 776, 781, 782, 785, 790, 791, 794, 795, 797, 800, 801, 805, 808, 809, 811, 812, 815, 819, 820, 824, 828, 831, 832, 836, 837, 838, 839, 841, 842, 847, 848, 849, 853, 859, 861, 863, 864, 868, 869, 870, 871, 874, 878, 880, 882, 883, 884, 886, 888, 890, 892, 897, 899, 901, 902, 903, 905, 906, 907, 908, 909, 911, 916, 917, 921, 923, 924, 925, 929, 931, 940, 941, 943, 946, 947, 950, 954, 955, 957, 958, 960, 962, 964, 965, 969, 971, 974, 976, 979, 980, 988, 989]
    gibson_better_ls = [65]
    hm3d_better_ls = [0, 1, 10, 11, 12, 14, 20, 34, 39, 41, 44, 45, 47, 50, 58, 64, 66, 74, 79, 84, 87, 97, 103, 105, 107, 117, 119, 126, 127, 128, 138, 139, 140, 142, 146, 152, 157, 158, 166, 170, 171, 172, 173, 184, 187, 191, 192, 198, 199, 205, 207, 213, 215, 223, 224, 227, 228, 231, 235, 241, 242, 244, 248, 252, 254, 255, 259, 261, 262, 263, 264, 275, 281, 286, 290, 291, 293, 294, 301, 302, 314, 323, 333, 337, 340, 341, 357, 366, 367, 370, 372, 374, 375, 378, 388, 404, 406, 408, 411, 412, 417, 420, 421, 427, 428, 429, 430, 435, 440, 442, 444, 449, 450, 453, 455, 459, 463, 465, 468, 476, 477, 482, 484, 497, 500, 503, 504, 505, 507, 525, 526, 529, 530, 534, 545, 546, 548, 550, 554, 555, 557, 565, 568, 570, 571, 572, 579, 581, 591, 594, 596, 599, 605, 606, 611, 612, 617, 618, 620, 621, 627, 628, 633, 634, 636, 638, 641, 646, 648, 649, 650, 660, 666, 667, 670, 673, 685, 686, 697, 706, 710, 714, 717, 729, 732, 738, 739, 740, 747, 752, 754, 758, 767, 772, 779, 780, 783, 784, 803, 804, 810, 813, 818, 821, 823, 826, 833, 834, 840, 843, 844, 845, 850, 852, 854, 855, 856, 857, 858, 860, 866, 867, 873, 875, 876, 881, 887, 889, 893, 895, 896, 900, 904, 912, 913, 919, 922, 926, 928, 933, 935, 936, 937, 938, 961, 968, 973, 975, 978, 981, 985, 986, 987, 993]

    for index_in_episodes in tqdm(range(args.graph_episode_num)):            
        # rl_graph_init
        rl_graph = RL_Graph()
        # haitat_episode_init
        observations = habitat_env.reset()
        # if(index_in_episodes<850):
        #     continue
        if(index_in_episodes not in gibson_better_ls):
            continue

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
            
            if(args.is_vis==True):
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=rl_graph, video_writer=video_writer, map_writer=map_writer)
            else:
                achieved_result = SubgoalReach.go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal)
            
            print("======> achieved_result <=====", achieved_result)
            print("=====> action_node_type <=====", action_node.node_type)

            rl_graph_action_node_name_ls = [(temp_node.name, temp_node.score) for temp_node in rl_graph.all_nodes if(temp_node.node_type!="explored_node")]
            print("=====> rl_graph_action_node_name_ls <=====", rl_graph_action_node_name_ls)
            print("============> intention_score <=============", action_node.score)
            
            
            evaluate_res = Evaluate.evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes)
            if(evaluate_res=="episode_stop"):
                break




