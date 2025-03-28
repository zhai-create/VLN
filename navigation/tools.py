import cv2
import numpy as np
import quaternion
from perception.arguments import args as perception_args
from graph.arguments import args as graph_args
from navigation.arguments import args
from graph.tools import get_absolute_pos

from env_tools.arguments import args as env_args
from navigation.RRTSTAR import RRTStar
from navigation.ASTAR import *

half_len = (int)(perception_args.graid_map_scale/graph_args.resolution)

def get_nearest_grid(end_point, temp_ghost_obstacle_map, action_category): # 寻找目标1m范围内最近的空闲位置
    """
        Get the nearest free area.
        :param end_point: frontier pos or intention pos
        :param temp_ghost_obstacle_map: map.
        :param action_category: action_node type.
        :return min_grid_x, min_grid_y: Nearest free map area.
    """
    if(action_category=="frontier_node"): # 要导航到frontier
        grid_delta = args.large_dis_thre
    else:
        grid_delta = args.small_dis_thre

    if(int(end_point[0])>=0 and int(end_point[0])<half_len and int(end_point[1])>=0 and int(end_point[1])<half_len):
        lower_bound_x = max(0, int(end_point[0]))
        lower_bound_y = max(0, int(end_point[1]))                                            
        upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end_point[0])+grid_delta)
        upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end_point[1])+grid_delta)
    elif (int(end_point[0])>=0 and int(end_point[0])<half_len and int(end_point[1])>=half_len and int(end_point[1])<2*half_len):
        lower_bound_x = max(0, int(end_point[0]))
        lower_bound_y = max(0, int(end_point[1])-grid_delta)
        upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end_point[0])+grid_delta)
        upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end_point[1]))
    elif (int(end_point[0])>=half_len and int(end_point[0])<2*half_len and int(end_point[1])>=0 and int(end_point[1])<half_len):
        lower_bound_x = max(0, int(end_point[0])-grid_delta)
        lower_bound_y = max(0, int(end_point[1]))
        upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end_point[0]))
        upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end_point[1])+grid_delta)
    else:
        lower_bound_x = max(0, int(end_point[0])-grid_delta)
        lower_bound_y = max(0, int(end_point[1])-grid_delta)
        upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end_point[0]))
        upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end_point[1]))

    min_grid_dis = -1
    min_grid_x = -1
    min_grid_y = -1

    for grid_x in range(lower_bound_x, upper_bound_x+1):
        for grid_y in range(lower_bound_y, upper_bound_y+1):
            if(temp_ghost_obstacle_map[grid_x][grid_y]<graph_args.unknown_val):
                temp_grid_dis = ((grid_x-int(end_point[0]))**2+(grid_y-int(end_point[1]))**2)**0.5
                if(temp_grid_dis<min_grid_dis or min_grid_dis<0):
                    min_grid_dis = temp_grid_dis
                    min_grid_x, min_grid_y = grid_x, grid_y
    return min_grid_x, min_grid_y

def get_l2_distance(x1, x2, y1, y2):
    """
        Get the euclidean distance.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_sim_location(habitat_env):
    """
        Returns x, y, o pose of the agent in the Habitat simulator.
    """
    agent_state = habitat_env._sim.get_agent_state(0)
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def get_rel_pose_change(pos2, pos1):
    """
        Get the pos change from pos1 and pos2.
    """
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1
    return dx, dy, do


def get_pose_change(habitat_env, last_sim_location):
    """
        Returns dx, dy, do pose change of the agent relative to the last timestep.
    """
    curr_sim_pose = get_sim_location(habitat_env)
    dx, dy, do = get_rel_pose_change(curr_sim_pose, last_sim_location)
    return dx, dy, do


def is_in_free_grid(temp_node, current_node, rela_cx, rela_cy):
    """
        Determine whether the current robot is located in the free area of temp_node.
        :param temp_node: Node to be inspected
        :param current_node
        :param rela_cx, rela_cy
        :return flag: True or False.
    """
    # temp_node是next_node,要将current_node坐标系下的位置转到temp_node下，看看在temp_node中是否可见
    if temp_node.name == current_node.name:
        rela_loc = np.array([rela_cx, rela_cy])
    else:
        current_node_in_temp_node = temp_node.all_other_nodes_loc[current_node.name] # 当前node在temp_node坐标系下的位置和角度
        rela_loc = get_absolute_pos(np.array([rela_cx, rela_cy]), current_node_in_temp_node[:2], current_node_in_temp_node[2])

    # 将相对于current_node坐标系下的位置转到了temp_node坐标系下，转换完毕

    rela_loc_t2 = rela_loc/graph_args.resolution
    rela_loc_p2 = np.array([-rela_loc_t2[0], rela_loc_t2[1]])
    rela_loc_end = rela_loc_p2+np.array([half_len, half_len])

    temp_node_obstacle_map = temp_node.occupancy_map[:,:,:]
    temp_node_obstacle_map = temp_node_obstacle_map[:,:,0]

    if(int(rela_loc_end[0])>=0 and int(rela_loc_end[0])<temp_node_obstacle_map.shape[0] and  int(rela_loc_end[1])>=0 and int(rela_loc_end[1])<temp_node_obstacle_map.shape[1]):
        if(temp_node_obstacle_map[int(rela_loc_end[0])][int(rela_loc_end[1])]<graph_args.unknown_val):
            return True
        else:
            return False
    else:
        return False


def plot_map(obstacles, path, suc):
    """
        Utility: Show the rrt path and its map.
    """
    if suc:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    thickness = 1
    point_size = 1

    gray_img = (obstacles * 255).astype(np.uint8)
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    if len(path) > 0:# coordinates of opencv is the reverse of oues.
        path_x = [int(p[1]) for p in path]  # [int(self.start.y), int(self.goal.y)]
        path_y = [int(p[0]) for p in path]  # [int(self.start.x), int(self.goal.x,)]
        for i in range(len(path_x)-1):
            cv2.line(rgb_img, (path_x[i], path_y[i]), (path_x[i+1], path_y[i+1]), color, thickness)
            cv2.circle(rgb_img, (path_x[i], path_y[i]), point_size, (255, 0, 0), thickness) # 起点：蓝点
            cv2.circle(rgb_img, (path_x[i+1], path_y[i+1]), point_size, (0, 255, 255), thickness) # 终点：黄色
    
    if(env_args.is_auto==False):
        rgb_img_for_show = cv2.resize(rgb_img, None, fx=1.5, fy=1.5)    
        cv2.imshow("RRT PATH", rgb_img_for_show) 
    # if(suc==False):
    #     rgb_img_for_show = cv2.resize(rgb_img, None, fx=1.5, fy=1.5)    
    #     cv2.imwrite("RRT_PATH.jpg", rgb_img_for_show) 

def is_temp_node_see(temp_node, current_node, rela_cx, rela_cy, rela_turn):
    # 判断current_node对应的rela_cx,rela_cy在temp_node下是否可见
    """
        Determine whether the center of the current node is within the range of the local grid of the temp_node.
    """
    if temp_node.name == current_node.name:
        rela_loc = np.array([rela_cx, rela_cy])
        rela_theta = rela_turn
    else:
        current_node_in_temp_node = temp_node.all_other_nodes_loc[current_node.name] # 当前node在temp_node坐标系下的位置和角度
        rela_loc = get_absolute_pos(np.array([rela_cx, rela_cy]), current_node_in_temp_node[:2], current_node_in_temp_node[2])
        rela_theta = rela_turn + current_node_in_temp_node[2]

    rela_loc_t2 = rela_loc/graph_args.resolution
    rela_loc_p2 = np.array([-rela_loc_t2[0], rela_loc_t2[1]])
    rela_loc_end = rela_loc_p2+np.array([100,100])

    temp_node_obstacle_map = temp_node.occupancy_map[:,:,:]
    temp_node_obstacle_map = temp_node_obstacle_map[:,:,0]

    if(int(rela_loc_end[0])>=0 and int(rela_loc_end[0])<temp_node_obstacle_map.shape[0] and  int(rela_loc_end[1])>=0 and int(rela_loc_end[1])<temp_node_obstacle_map.shape[1]):
        return True
    else:
        return False

def get_absolute_pos_world(rela_cx, rela_cy, world_cx, world_cy, world_turn):
    real_r_matrix = np.array([[np.cos(world_turn), np.sin(world_turn)], [np.sin(-world_turn), np.cos(world_turn)]])  
    res_loc_in_real_world = np.dot(real_r_matrix, np.array([rela_cx, rela_cy])) + np.array([-world_cx, world_cy])
    res_loc_in_real_world[0] = -res_loc_in_real_world[0]
    return res_loc_in_real_world


def get_relative_pos_world(real_world_cx, real_world_cy, world_cx, world_cy, world_turn):
    real_r_matrix = np.array([[np.cos(world_turn), -np.sin(world_turn)], [np.sin(world_turn), np.cos(world_turn)]])  
    rela_pos = np.dot(real_r_matrix, np.array([world_cx-real_world_cx, real_world_cy-world_cy]))
    return rela_pos

def get_a_star_path(sub_map, start_point, end_point):
    rrt = RRTStar(sub_map, start_point, end_point, inflation_distance=args.inflation_distance)
    if not (int(end_point[0])>=0 and int(end_point[0])<rrt.obstacles.shape[0] and int(end_point[1])>=0 and int(end_point[1])<rrt.obstacles.shape[1] and rrt.obstacles[int(end_point[0])][int(end_point[1])]<graph_args.unknown_val):
        # 如果在rrt地图中不能直接可见，则寻找其最近区域
        min_rrt_x, min_rrt_y = get_nearest_grid(end_point, temp_ghost_obstacle_map=rrt.obstacles, action_category="intention_node")
        if(min_rrt_x==-1): # 如果最近区域不可见
            zero_rrt_x, zero_rrt_y = get_nearest_grid(end_point, temp_ghost_obstacle_map=sub_map, action_category="intention_node")
            if(zero_rrt_x==-1):
                zero_rrt_x, zero_rrt_y = get_nearest_grid(end_point, temp_ghost_obstacle_map=sub_map, action_category="frontier_node")   
            if(zero_rrt_x!=-1): # 0311新加代码
                end_point[0], end_point[1] = zero_rrt_x, zero_rrt_y
            rrt = RRTStar(sub_map, start_point, end_point, inflation_distance=0) # 用膨胀因子为0的地图找
        else: # 如果最近区域可见
            end_point[0], end_point[1] = min_rrt_x, min_rrt_y
            rrt = RRTStar(sub_map, start_point, end_point, inflation_distance=args.inflation_distance)

    astar_map = Map(rrt.obstacles,int(start_point[0]),int(start_point[1]),int(end_point[0]),int(end_point[1]))
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


def get_node_robot_dis(rela_cx, rela_cy, sub_map, is_explored_node):
    near_dis = (rela_cx**2+rela_cy**2)**0.5
    if(near_dis<0.5):
        return near_dis
    
    # 使用A*算法计算实际路径长度
    end_point_row = (int)(half_len-rela_cx/graph_args.resolution)
    end_point_col = (int)(half_len+rela_cy/graph_args.resolution)
    end_point = np.array([end_point_row, end_point_col])
    start_point = np.array([int(half_len), int(half_len)])
    a_star_path = get_a_star_path(sub_map, start_point, end_point)
    if(a_star_path is not None):
        node_robot_dis = 0
        for temp_index, temp_loc in enumerate(a_star_path):
            if(temp_index==0):
                node_robot_dis += ((temp_loc[0]-start_point[0])**2+(temp_loc[1]-start_point[1])**2)**0.5
            else:
                node_robot_dis += ((temp_loc[0]-a_star_path[temp_index-1][0])**2+(temp_loc[1]-a_star_path[temp_index-1][1])**2)**0.5
        node_robot_dis = node_robot_dis*0.1 # 转换为meter为单位
    else:
        if(is_explored_node==True):
            node_robot_dis = ((rela_cx)**2+(rela_cy)**2)**0.5
        else:
            node_robot_dis = -1
    return node_robot_dis
    # 使用A*算法计算实际路径长度


def get_action_robot_dis(action_rela_cx, action_rela_cy, robot_rela_cx, robot_rela_cy, sub_map):
    if(abs(robot_rela_cx)<0.1 and abs(robot_rela_cy)<0.1):
        return (action_rela_cx**2+action_rela_cy**2)**0.5
    
    near_dis = ((robot_rela_cx-action_rela_cx)**2+(robot_rela_cy-action_rela_cy)**2)**0.5
    if(near_dis<0.5):
        return near_dis
    
    
    # 使用A*算法计算实际路径长度
    end_point_row = (int)(half_len-action_rela_cx/graph_args.resolution)
    end_point_col = (int)(half_len+action_rela_cy/graph_args.resolution)
    end_point = np.array([end_point_row, end_point_col])

    start_point_row = (int)(half_len-robot_rela_cx/graph_args.resolution)
    start_point_col = (int)(half_len+robot_rela_cy/graph_args.resolution)
    start_point = np.array([start_point_row, start_point_col])

    a_star_path = get_a_star_path(sub_map, start_point, end_point)
    if(a_star_path is not None):
        node_robot_dis = 0
        for temp_index, temp_loc in enumerate(a_star_path):
            if(temp_index==0):
                node_robot_dis += ((temp_loc[0]-start_point[0])**2+(temp_loc[1]-start_point[1])**2)**0.5
            else:
                node_robot_dis += ((temp_loc[0]-a_star_path[temp_index-1][0])**2+(temp_loc[1]-a_star_path[temp_index-1][1])**2)**0.5
        node_robot_dis = node_robot_dis*0.1 # 转换为meter为单位
    else:
        node_robot_dis = ((robot_rela_cx-action_rela_cx)**2+(robot_rela_cy-action_rela_cy)**2)**0.5
    return node_robot_dis
    # 使用A*算法计算实际路径长度