import copy
import numpy as np
from graph.tools import find_current_node, find_current_node_world, get_absolute_pos, clear_fake_frontier, get_current_world_pos, find_node_path
from graph.check_utils import second_check, third_check, forth_check
from graph.node_utils import Node
from graph.arguments import args
from perception.arguments import args as perception_args
from perception.frontier_utils import predict_frontier
from perception.tools import fix_depth, get_rgb_image_ls, get_gt_image_ls
from perception.intention_utils_rcnn import object_detect
# from perception.intention_utils_gt import object_detect_gt
from perception.intention_utils_gt_other import object_detect_gt_other
from perception.intention_utils_resnet import is_close

from env_tools.arguments import args as env_args
from navigation.tools import get_absolute_pos_world, get_relative_pos_world, get_a_star_path, get_node_robot_dis

if(env_args.is_llm==2):
    from perception.intention_utils_llava_easy_four_intention_tag import request_llm
elif(env_args.is_llm==1):
    from perception.intention_utils_llava_easy_three_intention_tag import request_llm
from perception.laser_utils import get_laser_point
from policy.rl_algorithms.arguments import args as rl_args

import copy
import time
from navigation.habitat_action import HabitatAction


half_len = (int)(perception_args.graid_map_scale/args.resolution)



class NodeList(list):
    def append(self, new_node):
        super().append(new_node)
        for existing_node in self:
            if existing_node.name != new_node.name:
                existing_node.add_other_node(new_node, self)
        for existing_node in self:
            if existing_node.name != new_node.name:
                new_node.add_other_node(existing_node, self)



class GraphMap(object):
    def __init__(self, habitat_env):
        self.explored_nodes = NodeList()
        self.explored_rotate_nodes = []
        self.frontier_nodes = []
        self.intention_nodes = []

        self.all_nodes = []

        self.current_node = None
        self.current_rotate_node = None


        self.rela_cx = 0
        self.rela_cy = 0
        self.rela_turn = 0

        self.point_for_close_loop_detection=None
        self.laser_2d_filtered=None
        self.laser_2d_filtered_angle=None
        self.pixel_y_2d_filtered = None
        
        self.habitat_env = habitat_env

        self.obs = None

        self.all_real_intentions = []
        self.obj_locations = [[] for i in range(21)] # 每用GLIP检测出一次，就记录一次 [[confidence, x, y], [confidence, x, y], ..., [confidence, x, y]]

    def set_current_pos(self, rela_cx, rela_cy, rela_turn):
        self.rela_cx = rela_cx
        self.rela_cy = rela_cy
        self.rela_turn = rela_turn

    def get_node(self, name):
        for temp_node in self.all_nodes:
            if(temp_node.name==name):
                return temp_node
    

    def frontier_delete(self): # 检查旧frontier在新地图上是否应该存在
        current_map = self.current_node.occupancy_map
        for temp_node in self.explored_nodes:            
            sub_frontiers_for_index = copy.deepcopy(temp_node.sub_frontiers)
            for temp_frontier in sub_frontiers_for_index: # 遍历所有的frontier
                # 先删除使得rl_step中实际行走的step为0的frontier
                # ===================================
                need_delete_flag = False
                for temp_delete_frontier in temp_node.deleted_frontiers:
                    if ((temp_delete_frontier.rela_cx-temp_frontier.rela_cx)**2+(temp_delete_frontier.rela_cy-temp_frontier.rela_cy)**2)**0.5<0.2:
                        need_delete_flag = True
                        break
                if(need_delete_flag == True):
                    temp_node.sub_frontiers.remove(temp_frontier)
                    self.frontier_nodes.remove(temp_frontier)
                    self.all_nodes.remove(temp_frontier)
                    continue
                # ===================================

                if temp_node.name != self.current_node.name:
                    n_in_current_node = self.current_node.all_other_nodes_loc[temp_node.name] # 将node中的ghost坐标位置转换到当前node下
                    g_ref_loc = get_absolute_pos(np.array([temp_frontier.rela_cx, temp_frontier.rela_cy]), n_in_current_node[:2], n_in_current_node[2])
                else:
                    g_ref_loc = np.array([temp_frontier.rela_cx, temp_frontier.rela_cy])
                gx = (int)(half_len-g_ref_loc[0]/args.resolution)
                gy = (int)(half_len+g_ref_loc[1]/args.resolution)

                if gx>=1 and gx<=(2*half_len-1) and gy>=1 and gy<=(2*half_len-1):
                    dis = ((g_ref_loc[0]-self.rela_cx)**2 + (g_ref_loc[1]-self.rela_cy)**2)**0.5
                    around = np.array([current_map[gx-1, gy-1, 0], current_map[gx-1, gy-0, 0], current_map[gx-1, gy+1, 0], current_map[gx-0, gy-1, 0], \
                                    current_map[gx-0, gy+1, 0], current_map[gx+1, gy-1, 0], current_map[gx+1, gy-0, 0], current_map[gx+1, gy+1, 0], current_map[gx, gy, 0]])
                    diff = np.absolute(around-args.ghost_map_g_val)
                    diff = np.sort(diff)
                    if diff[args.thre_for_delete] >= args.ghost_diff_thre or max(around) >= args.ghost_map_thre:
                        temp_node.sub_frontiers.remove(temp_frontier)
                        self.frontier_nodes.remove(temp_frontier)
                        self.all_nodes.remove(temp_frontier)

                    elif dis <= args.thre_for_blacklist_delete:
                        temp_node.sub_frontiers.remove(temp_frontier)
                        self.frontier_nodes.remove(temp_frontier)
                        self.all_nodes.remove(temp_frontier)
                        clear_fake_frontier(self.current_node, gx, gy)


    def multi_check_frontier(self, candidate_frontier_arr):
        res_frontier_pos_arr = []
        r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
        for index in range(candidate_frontier_arr.shape[0]):
            center_loc_in_ref = np.dot(r_matrix, candidate_frontier_arr[index]) + np.array([self.rela_cx, self.rela_cy])

            center_point_d1 = (int)(half_len-center_loc_in_ref[0]/args.resolution)
            center_point_d2 = (int)(half_len+center_loc_in_ref[1]/args.resolution)

            r = (int)((args.d_gap/args.resolution/np.sqrt(2)-1)/2)
            limit1 = min(center_point_d1-0, 2*half_len-center_point_d1)
            limit2 = min(center_point_d2-0, 2*half_len-center_point_d2)

            if r <= limit1-1 and r <= limit2-1: # 基于栅格状态的frontier_check
                gx, gy, second_flag = second_check(center_point_d1, center_point_d2, self.current_node.occupancy_map, r)
                if second_flag == False:
                    continue

                mx = -(gx - half_len) * args.resolution
                my = (gy - half_len) * args.resolution
                middle = np.array([mx, my]) # in ref frame (meter)
                third_flag = third_check(middle, self.current_node)
                if third_flag == False:
                    continue
                forth_flag = forth_check(middle, self.current_node, self.explored_nodes)
                if forth_flag == True:
                    res_frontier_pos_arr.append([middle[0], middle[1]])

        return np.array(res_frontier_pos_arr)

    def select_see_ghost(self, res_frontier_pos_arr):
        final_frontier_pos_arr = []

        temp_ghost_obstacle_map = self.current_node.occupancy_map[:,:,:]
        temp_ghost_obstacle_map = temp_ghost_obstacle_map[:,:,0]

        for index in range(res_frontier_pos_arr.shape[0]):
            temp_ghost_loc = res_frontier_pos_arr[index]
            ghost_t2 = temp_ghost_loc/args.resolution
            ghost_p2 = np.array([-ghost_t2[0], ghost_t2[1]])
            end = ghost_p2+np.array([half_len, half_len])

            if(temp_ghost_obstacle_map[int(end[0])][int(end[1])]<args.unknown_val):
                final_frontier_pos_arr.append(res_frontier_pos_arr[index])
                continue

            if(int(end[0])>=0 and int(end[0])<half_len and int(end[1])>=0 and int(end[1])<half_len):
                lower_bound_x = max(0, int(end[0]))
                lower_bound_y = max(0, int(end[1]))                                            
                upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end[0])+args.grid_delta)
                upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end[1])+args.grid_delta)
            elif (int(end[0])>=0 and int(end[0])<half_len and int(end[1])>=half_len and int(end[1])<2*half_len):
                lower_bound_x = max(0, int(end[0]))
                lower_bound_y = max(0, int(end[1])-args.grid_delta)
                upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end[0])+args.grid_delta)
                upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end[1]))
            elif (int(end[0])>=half_len and int(end[0])<2*half_len and int(end[1])>=0 and int(end[1])<half_len):
                lower_bound_x = max(0, int(end[0])-args.grid_delta)
                lower_bound_y = max(0, int(end[1]))
                upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end[0]))
                upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end[1])+args.grid_delta)
            else:
                lower_bound_x = max(0, int(end[0])-args.grid_delta)
                lower_bound_y = max(0, int(end[1])-args.grid_delta)
                upper_bound_x = min(temp_ghost_obstacle_map.shape[0]-1, int(end[0]))
                upper_bound_y = min(temp_ghost_obstacle_map.shape[1]-1, int(end[1]))
            
            
            flag1 = False
            for grid_x in range(lower_bound_x, upper_bound_x+1):
                flag2 = False
                for grid_y in range(lower_bound_y, upper_bound_y+1):
                    if(temp_ghost_obstacle_map[grid_x][grid_y]<args.unknown_val):
                        flag2 = True
                        break
                if(flag2==True):
                    flag1 = True
                    break
            if(flag1==True):
                final_frontier_pos_arr.append(res_frontier_pos_arr[index])
        return np.array(final_frontier_pos_arr)

    
    def add_ghost(self, final_frontier_pos_arr):
        for index in range(len(final_frontier_pos_arr)):
            res_loc_in_real_world = get_absolute_pos_world(final_frontier_pos_arr[index][0], final_frontier_pos_arr[index][1], self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
            

            new_frontier = Node(node_type="frontier_node", rela_cx=final_frontier_pos_arr[index][0], rela_cy=final_frontier_pos_arr[index][1], parent_node=self.current_node, world_cx=res_loc_in_real_world[0], world_cy=res_loc_in_real_world[1]) 
            self.current_node.sub_frontiers.append(new_frontier)
            self.frontier_nodes.append(new_frontier)
            self.all_nodes.append(new_frontier)

    def add_real_intention(self, real_res_pos_dict):
        rela_loc = np.array([self.rela_cx, self.rela_cy])
        r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
        
        for temp_score in real_res_pos_dict:
            for temp_rela_pos in real_res_pos_dict[temp_score]:
                tx, ty = temp_rela_pos[0], temp_rela_pos[1]
                center_loc_in_ref = np.dot(r_matrix, np.array([ty,tx])) + rela_loc
                res_loc_in_real_world = get_absolute_pos_world(center_loc_in_ref[0], center_loc_in_ref[1], self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
                self.all_real_intentions.append([res_loc_in_real_world[0], res_loc_in_real_world[1]])



    # def add_intention_gt(self, detect_res_pos_dict):
    #     rela_loc = np.array([self.rela_cx, self.rela_cy])
    #     r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
    
    #     new_intention_ls = []
    #     new_intention_name_ls = []
        
    #     world_cx, world_cy, world_cz, world_turn = get_current_world_pos(self.habitat_env)

    #     for temp_score in detect_res_pos_dict:
    #         for temp_rela_pos in detect_res_pos_dict[temp_score]:
    #             tx, ty = temp_rela_pos[0], temp_rela_pos[1]
    #             is_real_intention = temp_rela_pos[2]
    #             center_loc_in_ref = np.dot(r_matrix, np.array([ty,tx])) + rela_loc

    #             # cluster_revise
    #             res_loc_in_real_world = get_absolute_pos_world(center_loc_in_ref[0], center_loc_in_ref[1], self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
                
    #             # 防止fake_intention生成在real_intention附近(# intention_node聚类距离修改)
    #             if(is_real_intention==False):
    #                 is_near_real_flag = False
    #                 for temp_real_intention in self.intention_nodes:
    #                     if(temp_real_intention.is_real_intention==True):
    #                         if((res_loc_in_real_world[0]-temp_real_intention.world_cx)**2+(res_loc_in_real_world[1]-temp_real_intention.world_cy)**2)**0.5<1.0:
    #                             is_near_real_flag = True
    #                             break
    #                 if(is_near_real_flag==True):
    #                     continue
    #             # 防止fake_intention生成在real_intention附近
                
                
    #             new_intention = Node(node_type="intention_node", rela_cx=center_loc_in_ref[0], rela_cy=center_loc_in_ref[1], parent_node=self.current_node, score=temp_score, world_cx=res_loc_in_real_world[0], world_cy=res_loc_in_real_world[1], is_real_intention=is_real_intention)
    #             # correct_recheck
    #             new_intention.robot_intention_dis = ((new_intention.world_cx-world_cx)**2+(new_intention.world_cy-world_cy)**2)**0.5
    #             new_intention.dis_ls[0] = new_intention.robot_intention_dis
    #             new_intention.init_dis = new_intention.robot_intention_dis
    #             # correct_recheck
                
    #             self.current_node.sub_intentions.append(new_intention)
    #             self.intention_nodes.append(new_intention)
    #             self.all_nodes.append(new_intention)

    #             new_intention_ls.append(new_intention)
    #             new_intention_name_ls.append(new_intention.name)

    #     # correct_recheck
    #     for temp_intention_node in self.intention_nodes:
    #         if(temp_intention_node.name in new_intention_name_ls):
    #             continue
                
    #         min_dis = 10000
    #         min_node = None
    #         for temp_new_intention_node in new_intention_ls:
    #             temp_dis = ((temp_new_intention_node.world_cx-temp_intention_node.world_cx)**2+(temp_new_intention_node.world_cy-temp_intention_node.world_cy)**2)**0.5
    #             if(temp_dis<min_dis):
    #                 min_dis = temp_dis
    #                 min_node = temp_new_intention_node
    #         if(min_dis<=1.0): # intention_node聚类距离修改
    #             if(-1 not in temp_intention_node.score_ls):
    #                 temp_intention_node.score_ls.append(min_node.score)
    #                 temp_intention_node.score_ls.pop(0)

    #                 temp_intention_node.dis_ls.append(min_node.robot_intention_dis)
    #                 temp_intention_node.dis_ls.pop(0)
    #             else:
    #                 score_index = temp_intention_node.score_ls.index(-1)
    #                 temp_intention_node.score_ls[score_index] = min_node.score

    #                 temp_intention_node.dis_ls[score_index] = min_node.robot_intention_dis

    #             # 新增人工分数序列判断
    #             if(min_node.robot_intention_dis<temp_intention_node.init_dis):
    #                 temp_intention_node.near_score_ls.append(min_node.score)
    #     # correct_recheck

    def add_intention(self, detect_res_pos_dict):
        rela_loc = np.array([self.rela_cx, self.rela_cy])
        r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
    
        new_intention_ls = []
        new_intention_name_ls = []
        
        world_cx, world_cy, world_cz, world_turn = get_current_world_pos(self.habitat_env)

        for temp_score in detect_res_pos_dict:
            for temp_rela_pos in detect_res_pos_dict[temp_score]:
                tx, ty = temp_rela_pos[0], temp_rela_pos[1]
                center_loc_in_ref = np.dot(r_matrix, np.array([ty,tx])) + rela_loc

                # cluster_revise
                res_loc_in_real_world = get_absolute_pos_world(center_loc_in_ref[0], center_loc_in_ref[1], self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
                
                # # 防止fake_intention生成在real_intention附近(# intention_node聚类距离修改)
                # if(is_real_intention==False):
                #     is_near_real_flag = False
                #     for temp_real_intention in self.intention_nodes:
                #         if(temp_real_intention.is_real_intention==True):
                #             if((res_loc_in_real_world[0]-temp_real_intention.world_cx)**2+(res_loc_in_real_world[1]-temp_real_intention.world_cy)**2)**0.5<1.0:
                #                 is_near_real_flag = True
                #                 break
                #     if(is_near_real_flag==True):
                #         continue
                # # 防止fake_intention生成在real_intention附近
                
                
                new_intention = Node(node_type="intention_node", rela_cx=center_loc_in_ref[0], rela_cy=center_loc_in_ref[1], parent_node=self.current_node, score=temp_score, world_cx=res_loc_in_real_world[0], world_cy=res_loc_in_real_world[1])
                # correct_recheck
                new_intention.robot_intention_dis = ((new_intention.world_cx-world_cx)**2+(new_intention.world_cy-world_cy)**2)**0.5
                new_intention.dis_ls[0] = new_intention.robot_intention_dis
                new_intention.init_dis = new_intention.robot_intention_dis
                # correct_recheck
                
                self.current_node.sub_intentions.append(new_intention)
                self.intention_nodes.append(new_intention)
                self.all_nodes.append(new_intention)

                new_intention_ls.append(new_intention)
                new_intention_name_ls.append(new_intention.name)

        # correct_recheck
        for temp_intention_node in self.intention_nodes:
            if(temp_intention_node.name in new_intention_name_ls):
                continue
                
            min_dis = 10000
            min_node = None
            for temp_new_intention_node in new_intention_ls:
                temp_dis = ((temp_new_intention_node.world_cx-temp_intention_node.world_cx)**2+(temp_new_intention_node.world_cy-temp_intention_node.world_cy)**2)**0.5
                if(temp_dis<min_dis):
                    min_dis = temp_dis
                    min_node = temp_new_intention_node
            if(min_dis<=1.0): # intention_node聚类距离修改
                if(-1 not in temp_intention_node.score_ls):
                    temp_intention_node.score_ls.append(min_node.score)
                    temp_intention_node.score_ls.pop(0)

                    temp_intention_node.dis_ls.append(min_node.robot_intention_dis)
                    temp_intention_node.dis_ls.pop(0)
                else:
                    score_index = temp_intention_node.score_ls.index(-1)
                    temp_intention_node.score_ls[score_index] = min_node.score

                    temp_intention_node.dis_ls[score_index] = min_node.robot_intention_dis

                # 新增人工分数序列判断
                if(min_node.robot_intention_dis<temp_intention_node.init_dis):
                    temp_intention_node.near_score_ls.append(min_node.score)
        # correct_recheck

    def add_other_intention(self, other_res_pos_dict):
        rela_loc = np.array([self.rela_cx, self.rela_cy])
        r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
        
        world_cx, world_cy, world_cz, world_turn = get_current_world_pos(self.habitat_env)

        for temp_score in other_res_pos_dict:
            for temp_rela_pos in other_res_pos_dict[temp_score]:
                tx, ty = temp_rela_pos[0], temp_rela_pos[1]
                temp_object_text = temp_rela_pos[2]
                center_loc_in_ref = np.dot(r_matrix, np.array([ty,tx])) + rela_loc

                # cluster_revise
                res_loc_in_real_world = get_absolute_pos_world(center_loc_in_ref[0], center_loc_in_ref[1], self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
                self.obj_locations[HabitatAction.categories_21.index(temp_object_text)].append([temp_score, res_loc_in_real_world[0], res_loc_in_real_world[1]])

    def get_laser_result(self, depth):
        laser_2d_filtered, laser_2d_filtered_angle, pixel_y_2d_filtered = \
        get_laser_point(depth)

        # 表示最新的laser感知信息
        self.laser_2d_filtered = laser_2d_filtered
        self.laser_2d_filtered_angle = laser_2d_filtered_angle
        self.pixel_y_2d_filtered = pixel_y_2d_filtered
    
    def update(self):
        flag, predict_node, final_t, final_theta = find_current_node_world(self.explored_nodes, self.habitat_env, self.current_node)

        # update explored node
        if flag == True:
            last_node = self.current_node
            world_cx, world_cy, world_cz, world_turn = get_current_world_pos(self.habitat_env)
            
            if(last_node is None):
                self.set_current_pos(0.0, 0.0, 0.0)
                predict_node = Node(node_type="explored_node", world_cx=world_cx, world_cy=world_cy, world_cz=world_cz, world_turn=world_turn)
                self.current_node = predict_node
                self.all_nodes.append(self.current_node)
                self.explored_nodes.append(self.current_node)
            else:
                self.set_current_pos(0.0, 0.0, 0.0)
                predict_node = Node(node_type="explored_node", world_cx=world_cx, world_cy=world_cy, world_cz=world_cz, world_turn=world_turn)
                self.current_node = predict_node
                
                current_node_in_last_node_loc = get_relative_pos_world(self.current_node.world_cx, self.current_node.world_cy, last_node.world_cx, last_node.world_cy, last_node.world_turn)
                current_node_in_last_node_turn = self.current_node.world_turn-last_node.world_turn
                # node_robot_dis = get_node_robot_dis(current_node_in_last_node_loc[0], current_node_in_last_node_loc[1], last_node.occupancy_map, is_explored_node=True)

                last_node.add_neighbor(self.current_node, current_node_in_last_node_loc, current_node_in_last_node_turn)
                # last_node.neighbor_dis_dict[self.current_node.name] = node_robot_dis

                last_node_in_current_node_loc =  get_relative_pos_world(last_node.world_cx, last_node.world_cy, self.current_node.world_cx, self.current_node.world_cy, self.current_node.world_turn)
                last_node_in_current_node_turn = last_node.world_turn-self.current_node.world_turn
                self.current_node.add_neighbor(last_node, last_node_in_current_node_loc, last_node_in_current_node_turn)
                # self.current_node.neighbor_dis_dict[last_node.name] = node_robot_dis


                self.all_nodes.append(self.current_node)
                self.explored_nodes.append(self.current_node)

        elif flag == False and predict_node.name != self.current_node.name:
            self.set_current_pos(final_t[0], final_t[1], final_theta)

            last_node = self.current_node
            self.current_node = predict_node

            predicted_t_in_last = get_relative_pos_world(predict_node.world_cx, predict_node.world_cy, last_node.world_cx, last_node.world_cy, last_node.world_turn)
            predicted_theta_in_last = predict_node.world_turn-last_node.world_turn
            # node_robot_dis = get_node_robot_dis(predicted_t_in_last[0], predicted_t_in_last[1], last_node.occupancy_map, is_explored_node=True)

            last_node.add_neighbor(self.current_node, predicted_t_in_last, predicted_theta_in_last)
            # last_node.neighbor_dis_dict[self.current_node.name] = node_robot_dis

            last_t_in_predicted = get_relative_pos_world(last_node.world_cx, last_node.world_cy, predict_node.world_cx, predict_node.world_cy, predict_node.world_turn)
            last_theta_in_predicted = last_node.world_turn-predict_node.world_turn            
            self.current_node.add_neighbor(last_node, last_t_in_predicted, last_theta_in_predicted)     
            # self.current_node.neighbor_dis_dict[last_node.name] = node_robot_dis
        else:
            self.set_current_pos(final_t[0], final_t[1], final_theta)
        return flag


    def update_graph_frontier(self):
        candidate_frontier_arr = predict_frontier(args.init_predict_ghost_thre1, self.laser_2d_filtered, self.laser_2d_filtered_angle)
        res_frontier_pos_arr = self.multi_check_frontier(candidate_frontier_arr)
        final_frontier_pos_arr = self.select_see_ghost(res_frontier_pos_arr)
        self.add_ghost(final_frontier_pos_arr)
        self.frontier_delete()


    
    # 在决策之前，先判断action_pace是否为空？若为空，才进行该操作
    def ghost_patch(self, habitat_env, object_goal):
        depth = fix_depth(self.obs["depth"])
        self.get_laser_result(depth)
        predict_ghost_thre1 = args.init_predict_ghost_thre1
        while predict_ghost_thre1>=0:
            candidate_frontier_arr = predict_frontier(predict_ghost_thre1, self.laser_2d_filtered, self.laser_2d_filtered_angle)
            res_frontier_pos_arr = self.multi_check_frontier(candidate_frontier_arr)
            final_frontier_pos_arr = self.select_see_ghost(res_frontier_pos_arr)
            self.add_ghost(final_frontier_pos_arr)
            self.frontier_delete()
            if(len(self.frontier_nodes)>0):
                break
            else:
                predict_ghost_thre1 -= 0.1

        if(len(self.frontier_nodes)>0):
            return "ok" # 表示没有超过最大步数
        else: # 如果用当前点云进行阈值缩小后仍然没有找到合适的点云，则转圈
            for i in range(12):
                habitat_action = HabitatAction.set_habitat_action("r", self)
                if not habitat_env.episode_over:
                    observations = habitat_env.step(habitat_action)
                    print("======> patch <=====")
                    depth = fix_depth(observations["depth"])
                    self.get_laser_result(depth)
                    self.current_node.update_occupancy(self.laser_2d_filtered, self.laser_2d_filtered_angle, self.pixel_y_2d_filtered, np.array([self.rela_cx, self.rela_cy]), self.rela_turn)
                    predict_ghost_thre2 = args.init_predict_ghost_thre1
                    
                    while predict_ghost_thre2>=0:
                        candidate_frontier_arr = predict_frontier(predict_ghost_thre2, self.laser_2d_filtered, self.laser_2d_filtered_angle)
                        res_frontier_pos_arr = self.multi_check_frontier(candidate_frontier_arr)
                        final_frontier_pos_arr = self.select_see_ghost(res_frontier_pos_arr)
                        self.add_ghost(final_frontier_pos_arr)
                        self.frontier_delete()

                        if(len(self.frontier_nodes)>0):
                            break
                        else:
                            predict_ghost_thre2 -= 0.1

                    rgb_image_ls = get_rgb_image_ls(habitat_env)
                    gt_image_ls = get_gt_image_ls(habitat_env)
                    # detect_res_pos_dict = object_detect_gt(gt_image_ls, depth, object_goal, HabitatAction.object_id_num_ls)
                    # self.add_intention_gt(detect_res_pos_dict)

                    detect_res_pos_dict = object_detect(rgb_image_ls, depth, object_goal)
                    self.add_intention(detect_res_pos_dict)

                    other_res_pos_dict = object_detect_gt_other(gt_image_ls, depth, HabitatAction.other_object_id_num_ls)
                    self.add_other_intention(other_res_pos_dict)
                else:
                    return "exceed"
            return "ok" # 表示没有超过最大步数             


        


        


            
        

    
        
        


        


        

        