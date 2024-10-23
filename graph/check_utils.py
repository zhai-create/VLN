import numpy as np

from perception.arguments import args as perception_args
from graph.arguments import args
from graph.tools import get_absolute_pos


half_len = (int)(perception_args.depth_scale/args.resolution)


def second_check(cx, cy, map, r):
    ghost_exist = False
    frontier_exist = False
    flag = False
    dis = -1
    gx = 0
    gy = 0

    grid_row_lower = max(cx-r, 0)
    grid_row_upper = min(cx+r, map.shape[0]-1)

    grid_col_lower = max(cy-r, 0)
    grid_col_upper = min(cy+r, map.shape[1]-1)

    for temp_i in range(grid_row_lower, grid_row_upper+1):
        for temp_j in range(grid_col_lower, grid_col_upper+1):
            if np.absolute(map[temp_i, temp_j, 0] - args.ghost_map_g_val) < args.ghost_map_delta:
                around_ls = []
                for temp2_i in range(max(temp_i-1 ,0), min(temp_i+2, map.shape[0])):
                    for temp2_j in range(max(temp_j-1 ,0), min(temp_j+2, map.shape[1])):
                        if(temp2_i==temp_i and temp2_j==temp_j):
                            continue
                        around_ls.append(map[temp2_i, temp2_j, 0])
                around = np.array(around_ls)

                if max(around) <= args.ghost_map_thre and min(around) <= args.min_around_thre:
                    frontier_exist = True
                diff = np.absolute(around-args.ghost_map_g_val)
                diff = np.sort(diff)
                if diff[args.thre_for_detect] <= args.diff_upper:
                    ghost_exist = True
                    dis_t = (temp_i-cx)**2 + (temp_j-cy)**2
                    if (dis_t < dis) or (dis<0):
                        dis = dis_t
                        gx = temp_i
                        gy = temp_j

    if ghost_exist and frontier_exist:
        flag = True
    return gx, gy, flag

def third_check(middle, current_node):
    flag = True
    for temp_frontier in current_node.sub_frontiers:
        d = (middle[0]-temp_frontier.rela_cx)**2 + (middle[1]-temp_frontier.rela_cy)**2
        if d <= args.third_check_thre:
            flag = False
    return flag

def forth_check(middle, current_node, nodes_list):
    for temp_node in nodes_list:
        if temp_node.name == current_node.name:
            continue
        else:
            if temp_node.name == current_node.name:
                rela_loc = middle
            else:
                current_node_in_n = temp_node.all_other_nodes_loc[current_node.name]
                rela_loc = get_absolute_pos(middle, current_node_in_n[:2], current_node_in_n[2])
            
            gx = (int)(half_len-rela_loc[0]/args.resolution)
            gy = (int)(half_len+rela_loc[1]/args.resolution)
            if gx>=1 and gx<=(2*half_len-1) and gy>=1 and gy<=(2*half_len-1):
                temp_map = temp_node.occupancy_map
                around_ls = []
                for temp2_i in range(max(gx-1 ,0), min(gx+2, temp_map.shape[0])):
                    for temp2_j in range(max(gy-1 ,0), min(gy+2, temp_map.shape[1])):
                        around_ls.append(temp_map[temp2_i, temp2_j, 0])
                around = np.array(around_ls)
                
                diff = np.absolute(around-args.ghost_map_g_val)
                diff = np.sort(diff)
                if diff[args.thre_for_delete] >= args.ghost_diff_thre or max(around) >= args.ghost_map_thre:
                    return False
    return True