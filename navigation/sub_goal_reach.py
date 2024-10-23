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

class SubgoalReach:
    """
        Static class: Guide the robot to the sub-goal.
        Attributes
        ----------
        next_action: Selected next action.
        rotate_cnt: The number of times the robot shakes left and right.
        temp_action_buffer: Next action buffer.
        last_sim_location: last loc in the habitat.
        path_block_ls: List of robot walking path mater.

    """
    next_action = "new"
    rotate_cnt = 0
    temp_action_buffer = []
    last_sim_location = None
    path_block_ls = []

    @staticmethod
    def reset():
        """
            Reset the static attributes.
        """
        SubgoalReach.next_action = "new"
        SubgoalReach.rotate_cnt = 0
        SubgoalReach.temp_action_buffer = []
        SubgoalReach.last_sim_location = None
        SubgoalReach.path_block_ls = []

    @staticmethod
    def is_block(habitat_env):
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
            else:
                action_parent_node.sub_intentions.remove(action_node)
                topo_graph.intention_nodes.remove(action_node)
                topo_graph.all_nodes.remove(action_node)



    @staticmethod
    def go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal):
        """
            Go to the selected action node pos.
            :param topo_graph
            :param action_node
            :param habitat_env
            :param object_goal
        """
        topo_planner = TopoPlanner(topo_graph, action_node)
        SubgoalReach.reset()
        while True:
            if(HabitatAction.count_steps>env_args.max_steps): 
                return "exceed" # 判断是否在1m以内
            SubgoalReach.last_sim_location = get_sim_location(habitat_env)
            
            # 底层仿真器动作执行
            habitat_action = HabitatAction.set_habitat_action(SubgoalReach.next_action, topo_graph)
            if(habitat_action is None):
                continue

            if SubgoalReach.next_action=="f" or SubgoalReach.next_action=="l" or SubgoalReach.next_action=="r":
                observations = habitat_env.step(habitat_action)
                
                # 用于手动调试
                if(env_args.is_auto==False):
                    rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
                    cv2.imshow("rgb", rgb_image_ls[0])
                    cv2.imshow("depth", observations["depth"])
                
                if(SubgoalReach.next_action=="f"):
                    if(SubgoalReach.is_block(habitat_env)==True): # 认为自己卡住了，则跳出该函数，直接重新选择action node
                        # "block"
                        SubgoalReach.achieved_remove_action_node(topo_graph, action_node)
                        return "block" # new_patch1
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
                SubgoalReach.achieved_remove_action_node(topo_graph, action_node)
                return "achieved" # frontier: 继续选下一个action, intention: 判断是否在1m以内

            if(topo_planner.state_flag=="init" or (topo_planner.state_flag=="node_path" and SubgoalReach.next_action=="suc")):
                # topo_planner.get_topo_path()
                start_point, end_point, stitching_map = topo_planner.get_start_end_point()
                local_planner = LocalPlanner(stitching_map, start_point, end_point, topo_planner.state_flag, action_node, topo_graph, topo_planner.sub_map_node)
                local_path = local_planner.get_local_path()
                if(local_path is None):
                    # "Failed_Plan"
                    SubgoalReach.achieved_remove_action_node(topo_graph, action_node)
                    return "Failed_Plan" # new_patch2
            SubgoalReach.next_action, local_path, SubgoalReach.rotate_cnt, SubgoalReach.temp_action_buffer = local_planner.update_local_path(topo_planner, SubgoalReach.next_action, local_path, SubgoalReach.rotate_cnt, SubgoalReach.temp_action_buffer)
