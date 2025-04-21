import cv2
import numpy as np

from perception.arguments import args
from env_tools.data_utils import color_dict
from perception.tools import sam_show_mask, depth_estimation_object_loc

from navigation.habitat_action import HabitatAction



def object_detect_gt_other(gt_image_ls, depth, other_object_id_num_ls):
    other_res_pos_dict = {}  

    for temp_tuple in other_object_id_num_ls:
        temp_id_num, temp_object_text = temp_tuple[0], temp_tuple[1]
        new_mask = (gt_image_ls[0]==temp_id_num)[:, :, 0]
        true_count = np.sum(new_mask)

        

        if true_count < 50:
            continue
        else:
            res_depth_2d_cx, res_depth_2d_cy = depth_estimation_object_loc(new_mask, depth) # 相对于机器人的位姿
            if(res_depth_2d_cx is None) or ((res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5)<0.75:
                continue
            
            rule_dis = (res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5

            rule_score = -0.08*rule_dis+1
            if(rule_score>1):
                rule_score = 1
            elif(rule_score<0.6):
                rule_score = 0.6

            if(rule_score not in other_res_pos_dict):
                other_res_pos_dict[rule_score] = [[res_depth_2d_cx, res_depth_2d_cy, temp_object_text]]
            else:
                other_res_pos_dict[rule_score].append([res_depth_2d_cx, res_depth_2d_cy, temp_object_text])
    
    return other_res_pos_dict


print('Other perception initialize success!')







