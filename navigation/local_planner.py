import cv2
import copy
import numpy as np

from navigation.arguments import args
from graph.arguments import args as graph_args
from navigation.tools import get_nearest_grid, is_in_free_grid

from navigation.RRTSTAR import RRTStar
from navigation.ASTAR import *
from navigation.action_pub import choose_action

from graph.tools import get_absolute_pos
from env_tools.arguments import args as env_args

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


class LocalPlanner(object):
    """
        Plan on the local-map
    """
    def __init__(self, stitching_map, start_point, end_point, state_flag, action_node, topo_graph, sub_map_node, habitat_env):
        """
            Attributes
            ----------
            stitching_map: Map used for local planning.
            start_point: start_pos for local planning.
            end_point: end_pos for local planning.
            state_flag: Topo_planner state flag.
            action_node: selected sub-goal
            topo_graph
            sub_map_node: The node to which the submap belongs.
        """
        self.stitching_map = stitching_map
        self.start_point = start_point
        self.end_point = end_point


        self.state_flag = state_flag
        self.action_node = action_node
        self.topo_graph = topo_graph
        self.sub_map_node = sub_map_node

        self.habitat_env = habitat_env
        self.habitat_planner = ShortestPathFollower(self.habitat_env.sim, 0.5, False)


    def get_local_path(self): 
        """
            Get the local path based on astar
            :return local_path: local path
        """
        rrt = RRTStar(self.stitching_map, self.start_point, self.end_point, inflation_distance=args.inflation_distance)
        if not (int(self.end_point[0])>=0 and int(self.end_point[0])<rrt.obstacles.shape[0] and int(self.end_point[1])>=0 and int(self.end_point[1])<rrt.obstacles.shape[1] and rrt.obstacles[int(self.end_point[0])][int(self.end_point[1])]<graph_args.unknown_val):
            # 如果在rrt地图中不能直接可见，则寻找其最近区域
            min_rrt_x, min_rrt_y = get_nearest_grid(self.end_point, temp_ghost_obstacle_map=rrt.obstacles, action_category=self.action_node.node_type)
            if(min_rrt_x==-1): # 如果最近区域不可见
                zero_rrt_x, zero_rrt_y = get_nearest_grid(self.end_point, temp_ghost_obstacle_map=self.stitching_map, action_category=self.action_node.node_type)
                if(zero_rrt_x==-1):
                    zero_rrt_x, zero_rrt_y = get_nearest_grid(self.end_point, temp_ghost_obstacle_map=self.stitching_map, action_category="frontier_node")   
                self.end_point[0], self.end_point[1] = zero_rrt_x, zero_rrt_y
                rrt = RRTStar(self.stitching_map, self.start_point, self.end_point, inflation_distance=0) # 用膨胀因子为0的地图找
            else: # 如果最近区域可见
                self.end_point[0], self.end_point[1] = min_rrt_x, min_rrt_y
                rrt = RRTStar(self.stitching_map, self.start_point, self.end_point, inflation_distance=args.inflation_distance)

        astar_map = Map(rrt.obstacles,int(self.start_point[0]),int(self.start_point[1]),int(self.end_point[0]),int(self.end_point[1]))
        astar_path = astar(astar_map)
        if(astar_path != None): # 使用astar规划出来了路径
            local_path = astar_path
        else: # 没有使用astar规划出来路径，则使用腐蚀后的astar地图规划路径
            kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
            astar_map.data = cv2.erode(astar_map.data, kernel)
            astar_path = astar(astar_map)
            if(astar_path != None):
                local_path = astar_path
            else:
                local_path = None

        return local_path # 得到的local_path不包括起点


    def update_local_path(self, topo_planner, next_action, local_path):
        """
            Update local path and next action.
            :param topo_planner, next_action, local_path
            :return next_action, local_path
        """
        if self.sub_map_node.name == self.topo_graph.current_node.name:
            rela_loc = np.array([self.topo_graph.rela_cx, self.topo_graph.rela_cy])
            rela_theta = self.topo_graph.rela_turn
        else:
            current_node_in_end_node = self.sub_map_node.all_other_nodes_loc[self.topo_graph.current_node.name] # 当前node在rrt_node坐标系下的位置和角度
            rela_loc = get_absolute_pos(np.array([self.topo_graph.rela_cx, self.topo_graph.rela_cy]), current_node_in_end_node[:2], current_node_in_end_node[2])
            rela_theta = self.topo_graph.rela_turn + current_node_in_end_node[2]

        t1 = rela_loc / graph_args.resolution 
        p1 = np.array([-t1[0], t1[1]])
        self.start_point = p1+np.array([100,100]) # 单位：格，有小数，数组坐标。以current_rrt_node为参考系。
        

        if(len(local_path)==0):
            local_path = []
            next_action = "suc"
        elif(self.state_flag=="node_path" and is_in_free_grid(topo_planner.remain_nodes[0], self.topo_graph.current_node, self.topo_graph.rela_cx, self.topo_graph.rela_cy)==True):
            local_path = []
            next_action = "suc"
        else:
            local_path, next_action = choose_action(local_path, self.sub_map_node, self.habitat_env, self.habitat_planner)
        return next_action, local_path
            


    