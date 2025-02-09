import cv2
import numpy as np

import random

from perception.arguments import args
from env_tools.data_utils import color_dict
from perception.tools import sam_show_mask, depth_estimation, depth_estimation_laser, depth_estimation_laser_pinhole_to_panorama


def object_detect_gt_cluster(rgb_image_ls, depth, object_text, habitat_env):
    detect_res_pos_dict = {}    
    large_rgb = np.hstack((rgb_image_ls[2][:, int(rgb_image_ls[0].shape[1]//2):], rgb_image_ls[1], rgb_image_ls[0], rgb_image_ls[3], rgb_image_ls[2][:, :int(rgb_image_ls[0].shape[1]//2)]))

    # for hex_color in color_dict[habitat_env.episodes[0].scene_id][object_text]: # 遍历每一个样例
    for hex_color in color_dict[habitat_env.current_episode.scene_id][object_text]: # 遍历每一个样例
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        large_mask = np.all(large_rgb == rgb_color, axis=-1)

        true_count = np.sum(large_mask)
        if true_count < args.mask_true_cnt_thre:
            continue
        # if not np.any(large_mask): # 当前样例不存在于大图片中
        #     continue
        else:
            if(args.is_depth_estimation_laser==True):
                # res_depth_2d_cx, res_depth_2d_cy = depth_estimation_laser(large_mask, depth)
                res_depth_2d_cx, res_depth_2d_cy = depth_estimation_laser_pinhole_to_panorama(large_mask, depth)
            else:
                res_depth_2d_cx, res_depth_2d_cy = depth_estimation(large_mask, depth) # 相对于机器人的位姿

            if(res_depth_2d_cx is None):
                continue

            # cv2.imwrite("large_mask_{}_{}_{}.jpg".format(res_depth_2d_cx, res_depth_2d_cy, true_count), (large_mask*255).astype(np.uint8))
            
            rule_dis = (res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5
            rule_score = (0.2/(1+rule_dis**2))+0.8

            if(rule_score not in detect_res_pos_dict):
                detect_res_pos_dict[rule_score] = [[res_depth_2d_cx, res_depth_2d_cy]]
            else:
                detect_res_pos_dict[rule_score].append([res_depth_2d_cx, res_depth_2d_cy])
    
    
    
    other_object_ls = list(color_dict[habitat_env.current_episode.scene_id].keys())
    other_object_ls.remove(object_text)
    # 训练需要下面两行
    # random.shuffle(other_object_ls)
    # other_object_ls = other_object_ls[0:2]

    # for false_object in color_dict[habitat_env.current_episode.scene_id]: # 遍历其他错误类别
    for false_object in other_object_ls:
        if(false_object==object_text):
            continue
        for hex_color in color_dict[habitat_env.current_episode.scene_id][object_text]: # 遍历每一个错误类别的样例
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            large_mask = np.all(large_rgb == rgb_color, axis=-1)

            true_count = np.sum(large_mask)
            if true_count < args.mask_true_cnt_thre:
                continue
            # if not np.any(large_mask): # 当前样例不存在于大图片中
            #     continue
            else:
                if(args.is_depth_estimation_laser==True):
                    # res_depth_2d_cx, res_depth_2d_cy = depth_estimation_laser(large_mask, depth)
                    res_depth_2d_cx, res_depth_2d_cy = depth_estimation_laser_pinhole_to_panorama(large_mask, depth)
                else:
                    res_depth_2d_cx, res_depth_2d_cy = depth_estimation(large_mask, depth) # 相对于机器人的位姿

                if(res_depth_2d_cx is None):
                    continue

                rule_dis = (res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5
                rule_score = (0.1/(1+rule_dis**2))+0.6

                if(rule_score not in detect_res_pos_dict):
                    detect_res_pos_dict[rule_score] = [[res_depth_2d_cx, res_depth_2d_cy]]
                else:
                    detect_res_pos_dict[rule_score].append([res_depth_2d_cx, res_depth_2d_cy])


    return detect_res_pos_dict

print('GT perception initialize success!')


