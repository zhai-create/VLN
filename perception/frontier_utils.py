import numpy as np
from perception.arguments import args
from graph.arguments import args as graph_args

half_len = (int)(args.depth_scale/graph_args.resolution)

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

    for i in range(-1, laser_len):
        laser_dis = laser_2d_filtered[i]
        laser_angle = laser_2d_filtered_angle[i]
        tx = laser_dis * args.depth_scale * np.cos(laser_angle)
        ty = laser_dis * args.depth_scale * np.sin(laser_angle)
        laser_pos_ls.append(np.array([ty,tx]))

        if i >= 0 and np.absolute(laser_dis - laser_2d_filtered[(laser_len+i-1)%laser_len]) * args.depth_scale >= thre1 and laser_dis * args.depth_scale>= 0.01 and laser_2d_filtered[(laser_len+i-1)%laser_len] * args.depth_scale >= 0.01: 
            candidate_frontier_d1 = (ty + laser_pos_ls[-2][0])/2
            candidate_frontier_d2 = (tx + laser_pos_ls[-2][1])/2

            candidate_frontier_ls.append([candidate_frontier_d1, candidate_frontier_d2])

    return np.array(candidate_frontier_ls)