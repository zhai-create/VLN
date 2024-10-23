import cv2
import numpy as np
import quaternion
from perception.arguments import args as perception_args
from graph.arguments import args as graph_args
from navigation.arguments import args
from graph.tools import get_absolute_pos

from env_tools.arguments import args as env_args

half_len = (int)(perception_args.depth_scale/graph_args.resolution)

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


def plot_map(obstacles, path, suc, obstable_distance):
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

            # rgb_img = cv2.drawMarker(rgb_img, (path_x[i], path_y[i]), (255, 0, 0), cv2., thickness=1) # 起点：叉
            # rgb_img = cv2.drawMarker(rgb_img, (path_x[i+1], path_y[i+1]), (0, 255, 0), cv2.MARKER_SQUARE, thickness=1) # 终点：方块

    # if(suc==False):
    #     cv2.imwrite("train_rrt_21_one_modal_gibson_70_test_show/{}_{}_{}_{}_{}.jpg".format(len(os.listdir("train_rrt_21_one_modal_gibson_70_test_show/")), path[0][0], path[0][1], path[-1][0], path[-1][1]), rgb_img_for_show)
    # if(suc==False):
    #     np.save("train_rrt_21_one_modal_gibson_43_test_show/{}_{}_{}_{}_{}.npy".format(len(os.listdir("train_rrt_21_one_modal_gibson_43_test_show/")), path[0][0], path[0][1], path[-1][0], path[-1][1]), obstacles)
    
    if(env_args.is_auto==False):
        rgb_img_for_show = cv2.resize(rgb_img, None, fx=1.5, fy=1.5)    
        cv2.imshow("RRT PATH", rgb_img_for_show) # show00000000
    

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

