import numpy as np
from perception.arguments import args
from graph.arguments import args as graph_args

half_len = (int)(args.graid_map_scale/graph_args.resolution)

def find_longest_sequence(vec):
    is_five = (vec == 5)  # 找到所有等于 5 的位置，生成布尔数组

    # 使用 np.diff 和 np.where 找到连续 5 的起始和结束位置
    diff = np.diff(np.concatenate(([0], is_five.astype(int), [0])))

    

    starts = np.where(diff == 1)[0]  # 连续 5 的起始索引
    ends = np.where(diff == -1)[0] - 1  # 连续 5 的结束索引

    if len(starts) == 0:  # 如果没有连续 5 的子序列
        return None

    # 找到最长的连续 5 的子序列
    lengths = ends - starts + 1
    max_length_index = np.argmax(lengths)
    max_length = lengths[max_length_index]

    if max_length < 5:  # 如果最长子序列长度小于 5
        return None
    else:
        return starts[max_length_index], ends[max_length_index]


def predict_frontier(thre1, laser_2d_filtered, laser_2d_filtered_angle):
    """
    Get the candidate frontiers, which need the multicheck
    :param thre1: Gradient threshold for detection, which can be dynamically adjusted
    :param laser_2d_filtered: robot's current 2d-laser
    :param laser_2d_filtered_angle: corresponding angle of the 2d-laser
    :return candidate_frontier_ls: it needs the multicheck
    """
    
    
    laser_pos_ls = []
    candidate_frontier_ls = []
    laser_len = len(laser_2d_filtered)

    # for i in range(-1, laser_len):
    #     laser_dis = laser_2d_filtered[i]
    #     laser_angle = laser_2d_filtered_angle[i]
    #     tx = laser_dis * args.depth_scale * np.cos(laser_angle)
    #     ty = laser_dis * args.depth_scale * np.sin(laser_angle)
    #     laser_pos_ls.append(np.array([ty,tx]))

    #     if i >= 0 and np.absolute(laser_dis - laser_2d_filtered[(laser_len+i-1)%laser_len]) * args.depth_scale >= thre1 and laser_dis * args.depth_scale>= 0.01 and laser_2d_filtered[(laser_len+i-1)%laser_len] * args.depth_scale >= 0.01: 
    #         candidate_frontier_d1 = (ty + laser_pos_ls[-2][0])/2
    #         candidate_frontier_d2 = (tx + laser_pos_ls[-2][1])/2
    #         candidate_frontier_ls.append([candidate_frontier_d1, candidate_frontier_d2])

    for i in range(0, laser_len):
        laser_dis = laser_2d_filtered[i]
        laser_angle = laser_2d_filtered_angle[i]
        tx = laser_dis * args.depth_scale * np.cos(laser_angle)
        ty = laser_dis * args.depth_scale * np.sin(laser_angle)
        laser_pos_ls.append(np.array([ty,tx]))

        if i>=1 and np.absolute(laser_dis - laser_2d_filtered[i-1]) * args.depth_scale >= thre1: 
            if(laser_dis * args.depth_scale<1.0) or (laser_2d_filtered[i-1] * args.depth_scale<1.0):
                continue
            else:
                candidate_frontier_d1 = (ty + laser_pos_ls[-2][0])/2
                candidate_frontier_d2 = (tx + laser_pos_ls[-2][1])/2
                candidate_frontier_ls.append([candidate_frontier_d1, candidate_frontier_d2])



    find_res = find_longest_sequence(laser_2d_filtered*args.depth_scale)
    if(find_res is not None):
        start_index, end_index = find_res[0], find_res[1]
        mid_index = (start_index+end_index)//2
        laser_dis = laser_2d_filtered[mid_index]
        laser_angle = laser_2d_filtered_angle[mid_index]
        candidate_frontier_d1 = laser_dis * args.depth_scale * np.sin(laser_angle)
        candidate_frontier_d2 = laser_dis * args.depth_scale * np.cos(laser_angle)
        candidate_frontier_ls.append([candidate_frontier_d1, candidate_frontier_d2])

        
    return np.array(candidate_frontier_ls)
