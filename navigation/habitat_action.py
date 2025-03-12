import cv2
import numpy as np

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from navigation.arguments import args
from env_tools.arguments import args as env_args
from graph.tools import get_absolute_pos

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

    intention_one_cnt = 0

    episode_train_step = 0

    scene_file_dict = {}
    object_id_num_ls = []

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
                if(temp_object_name==object_text) or (temp_object_name=="tv"):
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
        HabitatAction.intention_one_cnt = 0
        HabitatAction.episode_train_step = 0
        HabitatAction.scene_file_dict = HabitatAction.get_current_scene_dict(habitat_env, graph_train)
        HabitatAction.object_id_num_ls = HabitatAction.get_object_num_ls(HabitatAction.scene_file_dict, object_text)


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

        