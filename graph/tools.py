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
half_len = (int)(perception_args.depth_scale/args.resolution)

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
def inverse_scanner(laser_2d_filtered, laser_2d_filtered_angle, relative_loc, relative_turn):
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
                    sub_map[i,j,0] = obstacle_val               
                elif r < laser_2d_filtered[k]*DEPTH_SCALE:
                    sub_map[i,j,0] = free_val
    return sub_map

def get_absolute_pos(p_loc, r_loc, rr):
    r_matrix = np.array([[np.cos(rr), np.sin(rr)], [-np.sin(rr), np.cos(rr)]])
    return r_loc + np.dot(r_matrix, p_loc)


def clear_fake_frontier(current_node, gx, gy):
    current_map = current_node.occupancy_map
    for i in range(args.clear_fake_lower, args.clear_fake_upper):
        for j in range(args.clear_fake_lower, args.clear_fake_upper):
            if np.absolute(current_map[gx+i, gy+j, 0] - args.ghost_map_g_val) <= args.ghost_map_delta:
                current_map[gx+i, gy+j, 0] = args.free_val

# RING
def find_current_node(explored_nodes, current_node, current_pc, rela_turn, rela_t):
    src_pc = current_pc
    I = Close_Loop()
    max_ratio = 0.0
    ratio_thre = 0.9
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
    return flag, pre_node, [final_theta, final_t], [theta_to_current, t_to_current], max_ratio


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


