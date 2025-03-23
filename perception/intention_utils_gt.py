import cv2
import numpy as np

from perception.arguments import args
from env_tools.data_utils import color_dict
from perception.tools import sam_show_mask, depth_estimation_object_loc

from navigation.habitat_action import HabitatAction



def object_detect_gt(gt_image_ls, depth, object_text, object_id_num_ls, is_fake_intention=False):
    detect_res_pos_dict = {}  

    for temp_id_num in object_id_num_ls:
        new_mask = (gt_image_ls[0]==temp_id_num)[:, :, 0]
        true_count = np.sum(new_mask)

        if true_count < 50:
            continue
        else:
            res_depth_2d_cx, res_depth_2d_cy = depth_estimation_object_loc(new_mask, depth) # 相对于机器人的位姿
            if(res_depth_2d_cx is None) or ((res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5)<0.75:
                continue
            
            # dis=1，mean=0.95, std=0.05
            # dis=5，mean=0.85，std=0.15

            rule_dis = (res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5
            rule_mean = (0.15/(1+0.25*(rule_dis**2)))+0.83
            rule_std = 0.025*rule_dis+0.025

            rule_score = HabitatAction.random_gen.normal(rule_mean, rule_std)

            if(rule_score>1):
                rule_score = 1
            elif(rule_score<0.6):
                rule_score = 0.6

            if(rule_score not in detect_res_pos_dict):
                detect_res_pos_dict[rule_score] = [[res_depth_2d_cx, res_depth_2d_cy, True]]
            else:
                detect_res_pos_dict[rule_score].append([res_depth_2d_cx, res_depth_2d_cy, True])




    # ==============> fake_intention <==============
    if(len(detect_res_pos_dict.keys())==0):
        if(HabitatAction.random_gen.uniform(0, 1)<0.3):
            is_generate_fake_flag = True
        else:
            is_generate_fake_flag = False
    else:
        if(HabitatAction.random_gen.uniform(0, 1)<0.1):
            is_generate_fake_flag = True
        else:
            is_generate_fake_flag = False
    
    if(is_generate_fake_flag==True) and (is_fake_intention==True):
        all_zero_matrix = np.zeros((120, 640))
        for temp_id_num in object_id_num_ls:
            num_mask = (gt_image_ls[0]==temp_id_num)[:, :, 0]
            all_zero_matrix += num_mask.astype(int)[180:300, :]
        
        all_bool_matrix = (all_zero_matrix==0)
        filtered_elements = gt_image_ls[0][180:300, :, 0][all_bool_matrix]

        if(filtered_elements.shape[0]!=0):
            unique_filtered_elements = np.unique(filtered_elements)[:HabitatAction.random_gen.randint(1, 3)]
            for temp_index in range(unique_filtered_elements.shape[0]):
                temp_id_num = unique_filtered_elements[temp_index]

                new_mask = (gt_image_ls[0]==temp_id_num)[:, :, 0]
                true_count = np.sum(new_mask)

                if true_count < 10:
                    continue
                else:
                    res_depth_2d_cx, res_depth_2d_cy = depth_estimation_object_loc(new_mask, depth) # 相对于机器人的位姿
                    if(res_depth_2d_cx is None) or ((res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5)<0.75:
                        continue


                    # dis=1，mean=0.65, std=0.05
                    # dis=5，mean=0.75，std=0.15

                    rule_dis = (res_depth_2d_cx**2+res_depth_2d_cy**2)**0.5
                    rule_mean = 0.025*rule_dis+0.625
                    rule_std = 0.025*rule_dis+0.025

                    rule_score = HabitatAction.random_gen.normal(rule_mean, rule_std)

                    if(rule_score>1):
                        rule_score = 1
                    elif(rule_score<0.6):
                        rule_score = 0.6

                    if(rule_score not in detect_res_pos_dict):
                        detect_res_pos_dict[rule_score] = [[res_depth_2d_cx, res_depth_2d_cy, False]]
                    else:
                        detect_res_pos_dict[rule_score].append([res_depth_2d_cx, res_depth_2d_cy, False])                
    # ==============> fake_intention <==============

    return detect_res_pos_dict


print('GT perception initialize success!')







