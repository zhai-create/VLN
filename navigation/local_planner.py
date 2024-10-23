import cv2
import copy
import numpy as np

from navigation.arguments import args
from graph.arguments import args as graph_args
from navigation.tools import get_nearest_grid, is_in_free_grid

from navigation.RRTSTAR import RRTStar
from navigation.ASTAR import *
from navigation.action_pub import choose_action, choose_action_origin

from graph.tools import get_absolute_pos
from env_tools.arguments import args as env_args


class LocalPlanner(object):
    """
        Plan on the local-map
    """
    def __init__(self, stitching_map, start_point, end_point, state_flag, action_node, topo_graph, sub_map_node):
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

    def get_local_path(self):
        """
            Get the local path based on rrt or astar
            :return rrt_path: local path
        """
        rrt = RRTStar(self.stitching_map, self.start_point, self.end_point, inflation_distance=args.inflation_distance)
        if(self.state_flag=="node_path"):
            rrt_path = rrt.planning()
        else:
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
            rrt_path = rrt.planning() # 先用rrt规划

        if(rrt_path is None): # 如果没有使用rrt规划出来路径
            astar_map = Map(rrt.obstacles,int(self.start_point[0]),int(self.start_point[1]),int(self.end_point[0]),int(self.end_point[1]))
            astar_path = astar(astar_map)
            if(astar_path != None): # 使用astar规划出来了路径
                astar_path.reverse()
                rrt_path = astar_path
            else: # 没有使用astar规划出来路径，则使用腐蚀后的astar地图规划路径
                kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
                astar_map.data = cv2.erode(astar_map.data, kernel)
                astar_path = astar(astar_map)
                if(astar_path != None):
                    astar_path.reverse()
                    rrt_path = astar_path
                else:
                    rrt_path = None
        
        if rrt_path is not None:
            rrt_path = rrt_path[1:]
        else:
            rrt_path = None
        return rrt_path

    def update_local_path(self, topo_planner, next_action, rrt_path, rotate_cnt, temp_action_buffer):
        """
            Update local path and next action.
            :param topo_planner, next_action, rrt_path, rotate_cnt, temp_action_buffer
            :return next_action, rrt_path, rotate_cnt, temp_action_buffer
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
        last_action = next_action
        if(len(rrt_path)==0):
            updated_path = []
            rrt_path = updated_path
            next_action = "suc"
            rotate_cnt = 0
            temp_action_buffer=[]
        
        elif(self.state_flag=="node_path" and is_in_free_grid(topo_planner.remain_nodes[0], self.topo_graph.current_node, self.topo_graph.rela_cx, self.topo_graph.rela_cy)==True):
            updated_path = []
            rrt_path = updated_path
            next_action = "suc"
            rotate_cnt = 0
            temp_action_buffer=[]
        else:
            if(args.is_choose_action==True):
                updated_path, next_action = choose_action(self.action_node.node_type, self.state_flag, last_action, rrt_path, (self.start_point[0], self.start_point[1]), (rela_theta)*180/np.pi+90)  
            else:
                updated_path, next_action = choose_action_origin(rrt_path, (self.start_point[0], self.start_point[1]), (rela_theta)*180/np.pi+90)  
            if((last_action=="l" or last_action=="r") and (next_action=="l" or next_action=="r") and (last_action!=next_action)):
                rotate_cnt += 1
            else:
                rotate_cnt = 0
            rrt_path = updated_path
            if(rotate_cnt >= args.max_rotate_cnt): # 暂时修改
                rrt_path = rrt_path[1:]
                rotate_cnt=0

            temp_action_buffer.append(next_action)
            if(len(temp_action_buffer)>args.len_action_buffer_thre):
                temp_action_buffer.pop(0)
                if(temp_action_buffer[0]=='l' and temp_action_buffer[1]=='l' and temp_action_buffer[2]=='r' and temp_action_buffer[3]=='r'):
                    rrt_path = rrt_path[1:]
                    temp_action_buffer=[]
        return next_action, rrt_path, rotate_cnt, temp_action_buffer
            


    