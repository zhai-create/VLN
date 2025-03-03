import numpy as np
from perception.arguments import args


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

            # # =====> laser_revise <=====    
            # if(laser_2d[i]>0.5): # 超过5m的激光进行mask
            #     laser_2d[i] = 0.01/args.depth_scale
            # # =====> laser_revise <=====   

            laser_2d_filtered.append(laser_2d[i])
            # temp_angle = 1.5 * np.pi - i / laser_len * 2 * np.pi
            temp_angle = 129.5*np.pi/180-(i/laser_len)*79*np.pi/180
            # if temp_angle >= np.pi:
            #     temp_angle = temp_angle - 2*np.pi
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
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    split_h = (int)(args.depth_height/2+1)
    intrinsic = args.intrinsic_matrix

    filter_z,filter_x = np.where(depth>-10) # 原始depth中大于0的位置
    depth_values_array = depth*args.depth_scale # meter
    
    filter_z_array = filter_z.reshape(args.depth_height, args.depth_width) # 行号矩阵    
    
    
    pixel_z = (depth.shape[0] - 1 - filter_z_array - intrinsic[1][2]) * depth_values_array / intrinsic[1][1]

    filter_x_array = filter_x.reshape(args.depth_height, args.depth_width) # 列号矩阵
    pixel_x = (filter_x_array - intrinsic[0][2])*depth_values_array / intrinsic[0][0]
    pixel_y = depth_values_array

    pixel_x = pixel_x[split_h:, :]
    pixel_y = pixel_y[split_h:, :]
    pixel_z = pixel_z[split_h:, :]

    pixel_z_array = np.expand_dims(pixel_z, axis=-1) # 全是负数

    laser_dis = (pixel_x**2+pixel_y**2)**0.5
    laser_dis_array = np.expand_dims(laser_dis, axis=-1)

    laser_points_for_noise_filter = np.concatenate((laser_dis_array, -pixel_z_array), axis = 2)


    laser_points_for_noise_filter[laser_points_for_noise_filter[:,:,1]>=args.camera_height+0.13-args.height_thre-0.3, 0] = 20.0
    laser_points_for_noise_filter[laser_points_for_noise_filter[:,:,1]<0.01, 0] = 20.0


    
    
    laser_row = np.argmin(laser_points_for_noise_filter[:,:,0], axis=0)
    laser_2d = laser_dis_array[laser_row, np.arange(args.depth_width), 0] / args.depth_scale # 0--1
    laser_2d_filtered, laser_2d_filtered_angle = laser_filter(laser_2d)

    
    # real_laser = np.concatenate((pixel_y.flatten().reshape(-1, 1), pixel_z.flatten().reshape(-1, 1), pixel_x.flatten().reshape(-1, 1)), axis=1)
    # real_high_light = np.concatenate((pixel_y[laser_row, np.arange(args.depth_width)].reshape(-1, 1), pixel_z[laser_row, np.arange(args.depth_width)].reshape(-1, 1), pixel_x[laser_row, np.arange(args.depth_width)].reshape(-1, 1)), axis=1)


    # breakpoint()
    return laser_2d_filtered, laser_2d_filtered_angle



# if __name__ == "__main__":
#     get_laser_point()
    