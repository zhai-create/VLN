import cv2
import numpy as np

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from navigation.arguments import args
from env_tools.arguments import args as env_args
from graph.tools import get_absolute_pos
from graph.arguments import args as graph_args
from perception.arguments import args as perception_args


import random
import numpy as np

class RandomGenerator:
    def __init__(self, seed):
        """
        初始化随机数生成器，设置随机数种子。
        """
        self.seed = seed
        random.seed(seed)  # 设置 Python random 模块的种子
        np.random.seed(seed)  # 设置 numpy 的随机数种子

    def random(self):
        """生成一个 [0, 1) 之间的随机浮点数。"""
        return random.random()

    def randint(self, low, high):
        """生成一个 [low, high] 之间的随机整数。"""
        return random.randint(low, high)

    def randn(self):
        """生成一个标准正态分布的随机数。"""
        return np.random.randn()

    def uniform(self, low, high):
        """生成一个 [low, high) 之间的均匀分布随机浮点数。"""
        return np.random.uniform(low, high)

    def normal(self, mean_val, std_val):
        return np.random.normal(loc=mean_val, scale=std_val)

    def reset(self, seed=None):
        """
        重置随机数种子。
        如果未提供 seed，则使用初始化时的种子。
        """
        if seed is not None:
            self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)


class HabitatAction:
    """
        static class for habitat action process.
        Attributes
        ----------
        count_steps: all step num.
        front_steps: front step num.
        walk_path_meter: The path meter of robot walking.
        this_episode_short_dis: init min_distance to goal.
    """
    count_steps = 0
    front_steps = 0
    walk_path_meter = 1e-5
    this_episode_short_dis = -1

    # train
    reward_per_episode = 0

    episode_train_step = 0

    scene_file_dict = {}
    object_id_num_ls = []

    init_all_map_loc = np.array([[500, 500]]) # size: (1001, 1001)

    name_val = 0
    real_intention_nodes = []

    random_gen = None

    fake_intention_check_flag = 0
    real_intention_check_flag = 0
    selected_intention_type_one_ls = []

    @staticmethod
    def get_current_scene_dict(habitat_env, graph_train):
        current_scene = habitat_env.current_episode.scene_id
        scene_num = current_scene.split('/')[-2].split("-")[0]
        scene_name = current_scene.split('/')[-2].split("-")[1]

        scene_file_dict = {} # key: object, value: id

        if(graph_train==True):
            open_file = "dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/{}-{}/{}.semantic.txt".format(scene_num, scene_name, scene_name)
        else:
            open_file = "dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/val/{}-{}/{}.semantic.txt".format(scene_num, scene_name, scene_name)

        with open(open_file, 'r') as file:
            next(file)  # 跳过第一行
            for line in file:
                columns = line.strip().split(',')
                temp_object_id = eval(columns[0])
                temp_object_name = columns[2].replace('"', '')
        
                if(temp_object_name not in scene_file_dict):
                    scene_file_dict[temp_object_name] = [temp_object_id]
                else:
                    scene_file_dict[temp_object_name].append(temp_object_id)
        return scene_file_dict

    @staticmethod
    def get_object_num_ls(scene_file_dict, object_text):
        object_id_num_ls = []
        
        if(object_text=="bed" or object_text=="toilet"):
            if(object_text in scene_file_dict):
                object_id_num_ls = scene_file_dict[object_text]
            else:
                object_id_num_ls = []
        
        elif(object_text=="sofa"):
            for temp_object_name in scene_file_dict:
                if(temp_object_name==object_text) or (temp_object_name=="couch"):
                    object_id_num_ls += scene_file_dict[temp_object_name]

        elif(object_text=="tv_monitor"):
            for temp_object_name in scene_file_dict:
                if(temp_object_name==object_text) or (temp_object_name=="tv") or (temp_object_name=="monitor") or (temp_object_name=="tv "):
                    object_id_num_ls += scene_file_dict[temp_object_name]

        else:
            for temp_object_name in scene_file_dict:
                if(object_text in temp_object_name):
                    object_id_num_ls += scene_file_dict[temp_object_name]

        return object_id_num_ls


    @staticmethod
    def reset(habitat_env, object_text, graph_train):
        """
            Reset the static attributes.
            :param habitat_env
        """
        HabitatAction.front_steps = 0
        HabitatAction.count_steps = 0
        HabitatAction.walk_path_meter = 1e-5
        HabitatAction.this_episode_short_dis = habitat_env.get_metrics()['distance_to_goal']
        
        HabitatAction.reward_per_episode = 0
        HabitatAction.episode_train_step = 0
        HabitatAction.scene_file_dict = HabitatAction.get_current_scene_dict(habitat_env, graph_train)
        HabitatAction.object_id_num_ls = HabitatAction.get_object_num_ls(HabitatAction.scene_file_dict, object_text)

        HabitatAction.init_all_map_loc = np.array([[500, 500]]) # size: (1001, 1001)s

        HabitatAction.name_val = 0
        HabitatAction.real_intention_nodes = []

        HabitatAction.random_gen = RandomGenerator(456)

        HabitatAction.fake_intention_check_flag = 0
        HabitatAction.real_intention_check_flag = 0
        HabitatAction.selected_intention_type_one_ls = []

    @staticmethod
    def get_all_map_loc(topo_graph):
        for temp_node in topo_graph.explored_nodes:
            if(temp_node.name == '0'):
                origin_row_col_indices = np.argwhere(temp_node.occupancy_map[:, :, 0] != graph_args.unknown_val)+400
                HabitatAction.init_all_map_loc = np.concatenate((HabitatAction.init_all_map_loc, origin_row_col_indices))
            else:
                zero_name_node = topo_graph.get_node("0")   
                temp_node_in_zero_node = zero_name_node.all_other_nodes_loc[temp_node.name]
                origin_row_col_indices = np.argwhere(temp_node.occupancy_map[:, :, 0] != graph_args.unknown_val)
                
                res_row_col_indices = []
                for temp_row_col_index in range(origin_row_col_indices.shape[0]):
                    row_index, column_index = origin_row_col_indices[temp_row_col_index][0], origin_row_col_indices[temp_row_col_index][1] 
                    rela_row_column_loc = get_absolute_pos(np.array([perception_args.graid_map_scale-graph_args.resolution*row_index, graph_args.resolution*column_index-perception_args.graid_map_scale]), temp_node_in_zero_node[:2], temp_node_in_zero_node[2])
                    row_column_t2 = rela_row_column_loc/graph_args.resolution
                    row_column_p2 = np.array([-row_column_t2[0], row_column_t2[1]])
                    row_column_loc = row_column_p2+np.array([500, 500])

                    new_row_index = round(row_column_loc[0])
                    new_column_index = round(row_column_loc[1])
                    res_row_col_indices.append([new_row_index, new_column_index])
                res_row_col_indices = np.array(res_row_col_indices)
                if(len(res_row_col_indices)>0):
                    HabitatAction.init_all_map_loc = np.concatenate((HabitatAction.init_all_map_loc, res_row_col_indices))
        HabitatAction.init_all_map_loc = np.unique(HabitatAction.init_all_map_loc, axis = 0)

    @staticmethod
    def get_all_see_intention(topo_graph, rl_graph):
        selected_intention_node_ls = rl_graph.select_intention(topo_graph)
        for temp_node in selected_intention_node_ls:
            if(temp_node.is_real_intention==True) and (temp_node not in HabitatAction.real_intention_nodes):
                HabitatAction.real_intention_nodes.append(temp_node)

    @staticmethod
    def set_habitat_action(action_name, topo_graph):
        """
            Process the habitat action and get the robot's pos after habitat_action.
            :param action_name: 'f' or 'l' or 'r'
            :param topo_graph
            :return habitat_action: action in the habitat_env.
        """
        if(env_args.is_auto==True):
            keystroke = ord(args.GO_ON)
        else:
            keystroke = cv2.waitKey(0)
        
        if keystroke == ord(args.GO_ON):
            if action_name == "f":
                habitat_action = HabitatSimActions.move_forward
                rela_cx, rela_cy = get_absolute_pos(np.array([args.forward_dis,0]), np.array([topo_graph.rela_cx, topo_graph.rela_cy]), topo_graph.rela_turn) # 计算前进1步后，机器人相对于当前node的坐标位置
                HabitatAction.front_steps += 1
                HabitatAction.count_steps += 1  # 每次初始化时增加步数
                topo_graph.set_current_pos(rela_cx, rela_cy, topo_graph.rela_turn)
                print("habitat_action: FORWARD")
            elif action_name == "l":
                habitat_action = HabitatSimActions.turn_left
                rela_turn = topo_graph.rela_turn + args.turn_angle / 180 * np.pi # 此处用topo_graph.rela_turn替换
                HabitatAction.count_steps += 1  # 每次初始化时增加步数
                topo_graph.set_current_pos(topo_graph.rela_cx, topo_graph.rela_cy, rela_turn)
                print("habitat_action: LEFT")
            elif action_name == "r":
                habitat_action = HabitatSimActions.turn_right
                rela_turn = topo_graph.rela_turn - args.turn_angle / 180 * np.pi
                HabitatAction.count_steps += 1  # 每次初始化时增加步数
                topo_graph.set_current_pos(topo_graph.rela_cx, topo_graph.rela_cy, rela_turn)
                print("habitat_action: RIGHT")
            elif action_name == "s": # 找到intention node后执行的动作(手动赋值)
                habitat_action = HabitatSimActions.stop
                HabitatAction.count_steps += 1  # 每次初始化时增加步数
                print("habitat_action: STOP")
            else: # "new" or "suc"
                habitat_action = -1
                print("Else_action:", action_name)
        elif keystroke == ord(args.FINISH): # 手动调试时才需要
            habitat_action = HabitatSimActions.stop
            HabitatAction.count_steps += 1  # 每次初始化时增加步数
            print("action: FINISH")
        else:
            habitat_action = None
            print("INVALID KEY")
        return habitat_action

        