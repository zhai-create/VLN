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

    @staticmethod
    def reset(habitat_env):
        """
            Reset the static attributes.
            :param habitat_env
        """
        HabitatAction.front_steps = 0
        HabitatAction.count_steps = 0
        HabitatAction.walk_path_meter = 1e-5
        HabitatAction.this_episode_short_dis = habitat_env.get_metrics()['distance_to_goal']
        
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

        