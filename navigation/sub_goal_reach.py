import cv2
import numpy as np
from env_tools.arguments import args as env_args


from navigation.arguments import args
from navigation.topo_planner import TopoPlanner
from navigation.local_planner import LocalPlanner
from navigation.habitat_action import HabitatAction
from navigation.tools import get_pose_change, get_sim_location, get_l2_distance


from habitat.sims.habitat_simulator.actions import HabitatSimActions

from perception.tools import get_rgb_image_ls, fix_depth
from vis_tools.vis_utils import save_mp4

import time

class SubgoalReach:
    """
        Static class: Guide the robot to the sub-goal.
        Attributes
        ----------
        next_action: Selected next action.
        last_sim_location: last loc in the habitat.
        path_block_ls: List of robot walking path mater.

    """
    next_action = "new"
    last_sim_location = None
    path_block_ls = []

    init_count_steps = 0
    init_front_steps = 0

    init_dis_to_goal = 0

    @staticmethod
    def reset(habitat_env):
        """
            Reset the static attributes.
        """
        SubgoalReach.next_action = "new"
        SubgoalReach.last_sim_location = None
        SubgoalReach.path_block_ls = []

        SubgoalReach.init_count_steps = HabitatAction.count_steps
        SubgoalReach.init_front_steps = HabitatAction.front_steps

        SubgoalReach.init_dis_to_goal = habitat_env.get_metrics()['distance_to_goal']

    @staticmethod
    def is_block(habitat_env, graph_train):
        """
            Determine if the robot is block.
            :param habitat_env
            :return flag: True or False
        """
        # 可能出现卡住滑动的现象
        dx, dy, do = get_pose_change(habitat_env, SubgoalReach.last_sim_location)
        HabitatAction.walk_path_meter += get_l2_distance(0, dx, 0, dy)  

        # 用于判断当前episode是否卡住
        SubgoalReach.path_block_ls.append(HabitatAction.walk_path_meter)  
        if(len(SubgoalReach.path_block_ls)>args.path_block_ls_thre):
            SubgoalReach.path_block_ls.pop(0)
            first_last_delta = abs(SubgoalReach.path_block_ls[-1]-SubgoalReach.path_block_ls[0])
            if(first_last_delta<=args.path_block_meter_thre):
                return True
        if(graph_train==True and (HabitatAction.front_steps-SubgoalReach.init_front_steps)>200):
            return True
        return False

    @staticmethod
    def achieved_remove_action_node(topo_graph, action_node):
        """
            When the robot achieves the sub-goal, delete the action node in the topo-graph.
            :param topo_graph
            :param action_node
        """
        if(action_node in topo_graph.all_nodes):
            action_parent_node = action_node.parent_node
            if(action_node.node_type=="frontier_node"):
                action_parent_node.sub_frontiers.remove(action_node)
                topo_graph.frontier_nodes.remove(action_node)
                topo_graph.all_nodes.remove(action_node)

                # rl_step中实际行走步数为0的frontier
                if((HabitatAction.count_steps-SubgoalReach.init_count_steps)==0):
                    action_parent_node.deleted_frontiers.append(action_node)


            else:
                action_parent_node.sub_intentions.remove(action_node)
                topo_graph.intention_nodes.remove(action_node)
                topo_graph.all_nodes.remove(action_node)


    def get_achieved_result(action_node, habitat_env, topo_graph, candidate_achieved_result, graph_train=False):
        if(graph_train==True): # 处于训练阶段，卡住直接退出
            if(action_node.node_type=="frontier_node" and candidate_achieved_result=="achieved"):
                SubgoalReach.achieved_remove_action_node(topo_graph, action_node)
                return candidate_achieved_result 
            else:
                if not habitat_env.episode_over:
                    habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                    observations = habitat_env.step(habitat_action)
                    return candidate_achieved_result
                else:
                    return "exceed"

        else: # 处于测试阶段
            if(action_node.node_type=="intention_node"):
                if not habitat_env.episode_over:
                    habitat_action = HabitatAction.set_habitat_action("s", topo_graph)
                    observations = habitat_env.step(habitat_action)
                    return candidate_achieved_result
                else:
                    return "exceed"
            else:
                SubgoalReach.achieved_remove_action_node(topo_graph, action_node)
                return candidate_achieved_result 



    @staticmethod
    def go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal, graph_train=False, rl_graph=None, video_writer=None, map_writer=None):
        """
            Go to the selected action node pos.
            :param topo_graph
            :param action_node
            :param habitat_env
            :param object_goal
        """
        topo_planner = TopoPlanner(topo_graph, action_node)
        SubgoalReach.reset(habitat_env)
        while True:
            # 执行动作前的位置检测
            SubgoalReach.last_sim_location = get_sim_location(habitat_env)
            
            # 底层仿真器动作执行
            habitat_action = HabitatAction.set_habitat_action(SubgoalReach.next_action, topo_graph)
            if(habitat_action is None): # 手动调试时INVALID KEY
                continue

            if (SubgoalReach.next_action=="f" or SubgoalReach.next_action=="l" or SubgoalReach.next_action=="r"):
                if not habitat_env.episode_over:
                    s_env_step_time = time.time()
                    observations = habitat_env.step(habitat_action)
                    e_env_step_time = time.time()
                    print("=====> delta_env_time <=====", e_env_step_time-s_env_step_time)
                else:
                    return "exceed"

                # 用于手动调试
                if(env_args.is_auto==False):
                    rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
                    cv2.imshow("rgb", rgb_image_ls[0])
                    cv2.imshow("depth", observations["depth"])
                
                # 用于录制视频
                if(env_args.is_vis==True):
                    save_mp4(video_writer, map_writer, habitat_env, topo_graph, rl_graph, action_node, object_goal)
                
                if(SubgoalReach.next_action=="f"):
                    if(SubgoalReach.is_block(habitat_env, graph_train)==True): # 认为自己卡住了，则跳出该函数，直接重新选择action node
                        # "block" # new_patch1
                        achieved_result = SubgoalReach.get_achieved_result(action_node, habitat_env, topo_graph, candidate_achieved_result="block", graph_train=graph_train)
                        return achieved_result
                    
                    else:
                        # get sensor data: rgb, depth, 2d_laser
                        rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
                        depth = fix_depth(observations["depth"])
                        topo_graph.update(rgb_image_ls, depth, object_goal)

                        # 用于手动调试
                        if(env_args.is_auto==False):
                            occu_for_show = cv2.resize(topo_graph.current_node.occupancy_map.astype(np.float64), None, fx=1, fy=1)
                            cv2.imshow("occu_for_show", occu_for_show)
            

            elif (SubgoalReach.next_action == "suc" and topo_planner.state_flag=="finish"):
                # "achieved"
                achieved_result = SubgoalReach.get_achieved_result(action_node, habitat_env, topo_graph, candidate_achieved_result="achieved", graph_train=graph_train)
                return achieved_result


            if(topo_planner.state_flag=="init" or (topo_planner.state_flag=="node_path" and SubgoalReach.next_action=="suc")):
                # topo_planner.get_topo_path()
                start_point, end_point, stitching_map = topo_planner.get_start_end_point()
                local_planner = LocalPlanner(stitching_map, start_point, end_point, topo_planner.state_flag, action_node, topo_graph, topo_planner.sub_map_node, habitat_env)
                local_path = local_planner.get_local_path()
                if(local_path is None):
                    # "Failed_Plan"
                    achieved_result = SubgoalReach.get_achieved_result(action_node, habitat_env, topo_graph, candidate_achieved_result="Failed_Plan", graph_train=graph_train)
                    return achieved_result

            SubgoalReach.next_action, local_path = local_planner.update_local_path(topo_planner, SubgoalReach.next_action, local_path)
            
            
            print("next_action:", SubgoalReach.next_action)
            print("topo_planner.state_flag:", topo_planner.state_flag)
            print("local_path:", local_path)