import math
from math import sqrt, atan2
from numba import jit
import numpy as np

from perception.arguments import args as perception_args
from graph.arguments import args
from graph.close_loop import Close_Loop

from collections import deque


beta = 2*np.pi/perception_args.depth_width * 2
alpha = 1*args.resolution
half_len = (int)(perception_args.graid_map_scale/args.resolution)

def fix_size(laser_2d_filtered, laser_2d_filtered_angle):
    origin_width = laser_2d_filtered.shape[0]
    laser_2d_filtered_res = np.zeros((perception_args.depth_width))
    laser_2d_filtered_angle_res = np.zeros((perception_args.depth_width))
    laser_2d_filtered_res[:origin_width] = laser_2d_filtered[:origin_width]
    laser_2d_filtered_res[origin_width:] = laser_2d_filtered[origin_width-1]
    laser_2d_filtered_angle_res[:origin_width] = laser_2d_filtered_angle[:origin_width]
    laser_2d_filtered_angle_res[origin_width:] = laser_2d_filtered_angle[origin_width-1]
    return laser_2d_filtered_res, laser_2d_filtered_angle_res




RESOLUTION = args.resolution
DEPTH_SCALE = perception_args.depth_scale
unknown_val = args.unknown_val
obstacle_val = args.obstacle_val
free_val = args.free_val
obstacle_dis = args.obstacle_dis

@jit(nopython=True)
def inverse_scanner(laser_2d_filtered, laser_2d_filtered_angle, pixel_y_2d_filtered, relative_loc, relative_turn):
    size = 2*half_len+1
    sub_map = np.ones((size, size, 1))/2
    x, y = (int)(half_len-relative_loc[0]/RESOLUTION), (int)(half_len+relative_loc[1]/RESOLUTION)
    for i in range(size):
        for j in range(size):
            if i == x and j == y:
                continue
            else:
                r = np.sqrt((i - x)**2 + (j - y)**2) * RESOLUTION #meter
                phi = (atan2(x-i, j-y) - relative_turn) % (2*np.pi)
                if phi > np.pi:
                    phi = phi - 2*np.pi

                difference1 = np.abs(np.subtract(phi, laser_2d_filtered_angle))
                k1 = np.argmin(difference1)
                difference2 = np.abs(np.subtract(np.abs(np.subtract(phi, laser_2d_filtered_angle)), 2*np.pi))
                k2 = np.argmin(difference2)
                if difference1[k1] <= difference2[k2]:
                    k = k1
                    diff = difference1[k1]
                else:
                    k = k2
                    diff = difference2[k2]

                if (r > laser_2d_filtered[k]*DEPTH_SCALE+alpha) or (diff > beta):
                    sub_map[i,j,0] = unknown_val
                elif np.abs(r-laser_2d_filtered[k]*DEPTH_SCALE) < alpha and laser_2d_filtered[k]*DEPTH_SCALE>obstacle_dis:
                    if(abs(pixel_y_2d_filtered[k]*DEPTH_SCALE-5)<=1e-3):
                        sub_map[i,j,0] = free_val    
                    else:
                        sub_map[i,j,0] = obstacle_val                 
                elif r < laser_2d_filtered[k]*DEPTH_SCALE:
                    sub_map[i,j,0] = free_val
    return sub_map

def get_absolute_pos(p_loc, r_loc, rr):
    r_matrix = np.array([[np.cos(rr), np.sin(rr)], [-np.sin(rr), np.cos(rr)]])
    return r_loc + np.dot(r_matrix, p_loc)

def get_relative_pos(f_loc, b_loc, rr):
    r_matrix = np.array([[np.cos(rr), -np.sin(rr)], [np.sin(rr), np.cos(rr)]])
    return np.dot(r_matrix, (f_loc-b_loc))


def clear_fake_frontier(current_node, gx, gy):
    current_map = current_node.occupancy_map
    for i in range(args.clear_fake_lower, args.clear_fake_upper):
        for j in range(args.clear_fake_lower, args.clear_fake_upper):
            if((gx+i)>=0 and (gx+i)<current_map.shape[0]) and ((gy+j)>=0 and (gy+j)<current_map.shape[1]):
                if np.absolute(current_map[gx+i, gy+j, 0] - args.ghost_map_g_val) <= args.ghost_map_delta:
                    current_map[gx+i, gy+j, 0] = args.free_val


def get_relative_pos_world(real_world_cx, real_world_cy, world_cx, world_cy, world_turn):
    real_r_matrix = np.array([[np.cos(world_turn), -np.sin(world_turn)], [np.sin(world_turn), np.cos(world_turn)]])  
    rela_pos = np.dot(real_r_matrix, np.array([world_cx-real_world_cx, real_world_cy-world_cy]))
    return rela_pos

# RING
def find_current_node(explored_nodes, current_node, current_pc, rela_turn, rela_t):
    src_pc = current_pc
    I = Close_Loop()
    max_ratio = 0.0
    # ratio_thre = 0.4
    ratio_thre = 0.5
    # ratio_thre = 0.9
    flag = True
    final_theta = None
    final_t = None
    theta_to_current = None
    t_to_current = None
    pre_node = None

    if(len(explored_nodes)==0):
        return True, pre_node, [final_theta, final_t], [theta_to_current, t_to_current], max_ratio

    for n in explored_nodes:
        if n.name == current_node.name:
            final_rela_turn = rela_turn
            final_rela_t = rela_t
        else:
            final_rela_turn = n.all_other_nodes_loc[current_node.name][2] + rela_turn
            current_in_n_node = n.all_other_nodes_loc[current_node.name] 
            rela_t_in_n = get_absolute_pos(rela_t, current_in_n_node[:2], current_in_n_node[2]) 
            final_rela_t = rela_t_in_n
        final_rela_t = final_rela_t / perception_args.depth_scale # 单位: meter --> 无量纲
        theta, t, matched_ratio= I.process(n.pc, src_pc, final_rela_turn, np.array([final_rela_t[1], final_rela_t[0], 0]))
        if matched_ratio > max_ratio:
            max_ratio = matched_ratio
            pre_node = n
            final_theta = theta
            final_t = np.array([t[1],t[0]]) * perception_args.depth_scale
        if n.name == current_node.name:
            theta_to_current = theta
            t_to_current = np.array([t[1],t[0]]) * perception_args.depth_scale
    if max_ratio <= ratio_thre:
        flag = True # generate new node
    else:
        flag = False # no generate
    
    print("==============================> max_ratio <==============================", max_ratio)
    
    return flag, pre_node, [final_theta, final_t], [theta_to_current, t_to_current], max_ratio


def find_current_node_world(explored_nodes, habitat_env, current_node):
    world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)

    if(len(explored_nodes)==0):
        flag = True
        pre_node = None
        rela_pos = np.array([0, 0])
        rela_turn = 0
    else:
        min_dis = 10000
        min_node = None
        for temp_explored_node in explored_nodes:
            temp_dis = ((world_cx-temp_explored_node.world_cx)**2+(world_cy-temp_explored_node.world_cy)**2)**0.5
            if(temp_dis<min_dis):
                min_dis = temp_dis
                min_node = temp_explored_node

        if(min_dis<4):
            flag = False
            pre_node = min_node
            rela_pos = get_relative_pos_world(world_cx, world_cy, min_node.world_cx, min_node.world_cy, min_node.world_turn)
            rela_turn = world_turn-min_node.world_turn
        
            pre_node_obstacle_map = pre_node.occupancy_map[:,:,0]
            current_rela_loc = np.array([rela_pos[0], rela_pos[1]])
            object_t2 = current_rela_loc/args.resolution
            object_p2 = np.array([-object_t2[0], object_t2[1]])
            current_grid_pos = object_p2+np.array([half_len, half_len])
            if(int(current_grid_pos[0])>=0 and int(current_grid_pos[0])<pre_node_obstacle_map.shape[0] and int(current_grid_pos[1])>=0 and int(current_grid_pos[1])<pre_node_obstacle_map.shape[1]):
                if(pre_node_obstacle_map[int(current_grid_pos[0])][int(current_grid_pos[1])]<args.unknown_val):
                    flag = False
                else:
                    if(((world_cx-current_node.world_cx)**2+(world_cy-current_node.world_cy)**2)**0.5<7):
                        flag = False
                        pre_node = current_node
                        rela_pos = get_relative_pos_world(world_cx, world_cy, current_node.world_cx, current_node.world_cy, current_node.world_turn)
                        rela_turn = world_turn-current_node.world_turn
                    else:
                        flag = True
                        pre_node = None
                        rela_pos = np.array([0, 0])
                        rela_turn = 0
            else:
                flag = True
                pre_node = None
                rela_pos = np.array([0, 0])
                rela_turn = 0

        else:
            flag = True
            pre_node = None
            rela_pos = np.array([0, 0])
            rela_turn = 0
    return flag, pre_node, rela_pos, rela_turn


def find_node_path(n1, n2, explored_nodes):
    if n1.name == n2.name:
        return [n1]

    visited = set()
    queue = deque([[n1]])

    while queue:
        path = queue.popleft() # [n1]
        node = path[-1] # n1

        if node.name in visited:
            continue

        for neighbor_name in node.neighbor:
            neighbor = next((n for n in explored_nodes if n.name == neighbor_name), None)

            if neighbor:
                new_path = list(path)
                new_path.append(neighbor)

                if neighbor.name == n2.name:
                    return new_path

                queue.append(new_path)

        visited.add(node.name)
    return None


def get_current_world_pos(habitat_env):
    world_cx, world_cy = habitat_env._sim.get_agent_state(0).position[2], habitat_env._sim.get_agent_state(0).position[0]
    world_cz = habitat_env._sim.get_agent_state(0).position[1]
    world_turn = 2 * math.atan(habitat_env._sim.get_agent_state(0).rotation.y/habitat_env._sim.get_agent_state(0).rotation.w)
    return world_cx, world_cy, world_cz, world_turn


def get_min_goal_loc(current_episode, world_cx, world_cy):
    # 找到距离当前机器人最近的object所在的位置
    min_goal_dis = 1000000
    min_goal_loc = None
    for temp_index in range(len(current_episode.goals)):
        temp_dis = ((current_episode.goals[temp_index].position[2]-world_cx)**2+(current_episode.goals[temp_index].position[0]-world_cy)**2)**0.5
        if temp_dis<min_goal_dis:
            min_goal_dis = temp_dis
            min_goal_loc = current_episode.goals[temp_index].position
    return min_goal_loc

