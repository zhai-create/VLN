import numpy as np
from perception.arguments import args, pre_depth


def laser_filter(laser_2d):
    """
    Filter the 2d-laser and get corresponding angle
    :param laser_2d: original laser
    :return laser_2d_filtered:  filtered 2d-laser based on gradient 
    :return laser_2d_filtered_angle: corresponding angle of the 2d-laser
    """
    laser_2d_filtered = []
    laser_2d_filtered_angle = []

    laser_len = len(laser_2d)
    for i in range(laser_len):
        left_ave_depth_dif = (np.absolute(laser_2d[(i-1+laser_len)%laser_len]-laser_2d[(i-2+laser_len)%laser_len])*args.depth_scale \
                            + np.absolute(laser_2d[(i-2+laser_len)%laser_len]-laser_2d[(i-3+laser_len)%laser_len])*args.depth_scale \
                            + np.absolute(laser_2d[(i-3+laser_len)%laser_len]-laser_2d[(i-4+laser_len)%laser_len])*args.depth_scale) / 3
        right_ave_depth_dif = (np.absolute(laser_2d[(i+1+laser_len)%laser_len]-laser_2d[(i+2+laser_len)%laser_len])*args.depth_scale \
                            + np.absolute(laser_2d[(i+2+laser_len)%laser_len]-laser_2d[(i+3+laser_len)%laser_len])*args.depth_scale \
                            + np.absolute(laser_2d[(i+3+laser_len)%laser_len]-laser_2d[(i+4+laser_len)%laser_len])*args.depth_scale) / 3  


        if (np.absolute(laser_2d[i]-laser_2d[(i-1+laser_len)%laser_len]) * args.depth_scale < min(args.filter_thre, 2.5*left_ave_depth_dif) \
        or np.absolute(laser_2d[i]-laser_2d[(i+1+laser_len)%laser_len]) * args.depth_scale < min(args.filter_thre, 2.5*right_ave_depth_dif)) \
        or laser_2d[i]*args.depth_scale < 0.01: 
            laser_2d_filtered.append(laser_2d[i])
            temp_angle = 1.5 * np.pi - i / laser_len * 2 * np.pi
            if temp_angle >= np.pi:
                temp_angle = temp_angle - 2*np.pi
            laser_2d_filtered_angle.append(temp_angle)
    laser_2d_filtered = np.array(laser_2d_filtered)
    laser_2d_filtered_angle = np.array(laser_2d_filtered_angle)
    return laser_2d_filtered, laser_2d_filtered_angle


def get_laser_point(depth):
    """
    Get the filtered 2d-laser
    :param depth: depth after fixed
    :return point_for_close_loop_detection: 2d-laser for ring
    :return laser_2d_filtered:  robot's current 2d-laser
    :return laser_2d_filtered_angle: corresponding angle of the 2d-laser
    """
    split_h = (int)(args.depth_height/2+1)
    split_w = (int)(args.depth_width/2+1)

    depth_half = depth[split_h:, :, :]
    laser_height = -depth_half * np.sin(pre_depth.data[split_h:, :, 2:3]) * args.depth_scale # meter
    laser_dis = depth_half * np.cos(pre_depth.data[split_h:, :, 2:3]) * args.depth_scale # meters
    laser_angle = -pre_depth.data[split_h:, :, 3:4] + 0.5 * np.pi
    laser_x = laser_dis * np.sin(laser_angle) # meter
    laser_z = laser_dis * np.cos(laser_angle)
    laser_points = np.concatenate((laser_x, laser_height, laser_z), axis = 2)

    laser_points_for_noise_filter = np.concatenate((laser_dis, -laser_height), axis = 2)
    laser_points_for_noise_filter[laser_points_for_noise_filter[:,:,1]>=args.camera_height+0.13-args.height_thre, 0] = 20.0
    laser_row = np.argmin(laser_points_for_noise_filter[:,:,0], axis=0)
    laser_2d = laser_dis[laser_row, np.arange(args.depth_width), 0] / args.depth_scale # 0--1

    laser_points = laser_points.reshape(-1,3)
    laser_points[:,[0,1,2]] = laser_points[:,[2,0,1]]
    laser_points = laser_points.astype(np.float16)

    laser_2d_filtered, laser_2d_filtered_angle = laser_filter(laser_2d)

    depth_for_unprojection_for_close_loop = depth[split_h-1:split_h, :, :]
    laser_dis_for_close_loop = depth_for_unprojection_for_close_loop * np.cos(pre_depth.data[split_h:, :, 2:3]) * args.depth_scale # 单位: meter
    laser_for_close_loop = laser_dis_for_close_loop[0, np.arange(args.depth_width), 0] / args.depth_scale # 单位: 无量纲

    laser_2d_filtered_for_close_loop, laser_2d_filtered_angle_for_close_loop = laser_filter(laser_for_close_loop) # 深度相机中间高度的filter_scan和filter_scan_angle

    number_of_filtered = len(laser_2d_filtered_for_close_loop)
    indices = np.arange(0, number_of_filtered, args.index_ratio) 
    new_laser_2d_filtered = laser_2d_filtered_for_close_loop[indices]
    new_laser_2d_filtered_angle = laser_2d_filtered_angle_for_close_loop[indices]
    x_for_close_loop = new_laser_2d_filtered * np.cos(new_laser_2d_filtered_angle)
    y_for_close_loop = new_laser_2d_filtered * np.sin(new_laser_2d_filtered_angle)
    z_for_close_loop = np.ones(number_of_filtered)
    point_for_close_loop_detection = np.array([x_for_close_loop, y_for_close_loop, z_for_close_loop])
    return point_for_close_loop_detection, laser_2d_filtered, laser_2d_filtered_angle




# if __name__ == "__main__":
#     get_laser_point()
    
