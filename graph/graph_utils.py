import copy
import numpy as np
from graph.tools import find_current_node, get_absolute_pos, clear_fake_frontier, get_current_world_pos
from graph.check_utils import second_check, third_check, forth_check
from graph.node_utils import Node
from graph.arguments import args
from perception.arguments import args as perception_args
from perception.frontier_utils import predict_frontier
from perception.intention_utils_rcnn import object_detect
from perception.intention_utils_blip import request_llm
from perception.laser_utils import get_laser_point

import copy
import time



half_len = (int)(perception_args.depth_scale/args.resolution)



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
        self.frontier_nodes = []
        self.intention_nodes = []
        self.all_nodes = []

        self.current_node = None
        self.rela_cx = 0
        self.rela_cy = 0
        self.rela_turn = 0

        self.point_for_close_loop_detection=None
        self.laser_2d_filtered=None
        self.laser_2d_filtered_angle=None
        
        self.habitat_env = habitat_env


    def set_current_pos(self, rela_cx, rela_cy, rela_turn):
        self.rela_cx = rela_cx
        self.rela_cy = rela_cy
        self.rela_turn = rela_turn

    def get_node(self, name):
        for temp_node in self.all_nodes:
            if(temp_node.name==name):
                return temp_node
    
    
    def frontier_delete(self):
        show_ghost_map = copy.deepcopy(self.current_node.occupancy_map)
        show_ghost_map[show_ghost_map>=args.ghost_map_thre] = 1
        show_ghost_map[show_ghost_map<args.ghost_map_thre] = 0

        current_map = self.current_node.occupancy_map

        for temp_node in self.explored_nodes:            
            sub_frontiers_for_index = copy.deepcopy(temp_node.sub_frontiers)
            for temp_frontier in sub_frontiers_for_index:
                

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
                

                temp_dis = ((g_ref_loc[0]-self.rela_cx)**2 + (g_ref_loc[1]-self.rela_cy)**2)**0.5

                if gx>=1 and gx<=(2*half_len-1) and gy>=1 and gy<=(2*half_len-1):
                    temp_val = show_ghost_map[gx,gy,0]
                    show_ghost_map[gx,gy,0] = args.ghost_map_g_val
                    dis = ((g_ref_loc[0]-self.rela_cx)**2 + (g_ref_loc[1]-self.rela_cy)**2)**0.5
                    
                    around = np.array([current_map[gx-1, gy-1, 0], current_map[gx-1, gy-0, 0], current_map[gx-1, gy+1, 0], current_map[gx-0, gy-1, 0], \
                                    current_map[gx-0, gy+1, 0], current_map[gx+1, gy-1, 0], current_map[gx+1, gy-0, 0], current_map[gx+1, gy+1, 0], current_map[gx, gy, 0]])
                    diff = np.absolute(around-args.ghost_map_g_val)
                    diff = np.sort(diff)
                    if diff[args.thre_for_delete] >= args.ghost_diff_thre or max(around) >= args.ghost_map_thre:
                        temp_node.sub_frontiers.remove(temp_frontier)
                        self.frontier_nodes.remove(temp_frontier)
                        self.all_nodes.remove(temp_frontier)
                        show_ghost_map[gx,gy,0] = temp_val
                    elif dis <= args.thre_for_blacklist_delete:
                        temp_node.sub_frontiers.remove(temp_frontier)
                        self.frontier_nodes.remove(temp_frontier)
                        self.all_nodes.remove(temp_frontier)
                        clear_fake_frontier(self.current_node, gx, gy)
                else: # 直接remove
                    temp_node.sub_frontiers.remove(temp_frontier)
                    self.frontier_nodes.remove(temp_frontier)
                    self.all_nodes.remove(temp_frontier)


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

            first_flag = False
            second_flag = False
            if r <= limit1-1 and r <= limit2-1:
                gx, gy, second_flag = second_check(center_point_d1, center_point_d2, self.current_node.occupancy_map, r)
            if second_flag == False:
                continue
            else:
                mx = -(gx - half_len) * args.resolution
                my = (gy - half_len) * args.resolution
                middle = np.array([mx, my]) # in ref frame (meter)
                third_flag = third_check(middle, self.current_node)

            if third_flag == False:
                continue
            else:
                forth_flag = forth_check(middle, self.current_node, self.explored_nodes)
            if forth_flag == True:
                first_flag = True

            if first_flag == True:
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
            new_frontier = Node(node_type="frontier_node", rela_cx=final_frontier_pos_arr[index][0], rela_cy=final_frontier_pos_arr[index][1], parent_node=self.current_node)
            self.current_node.sub_frontiers.append(new_frontier)
            self.frontier_nodes.append(new_frontier)
            self.all_nodes.append(new_frontier)

    def add_intention(self, detect_res_pos_dict):
        rela_loc = np.array([self.rela_cx, self.rela_cy])
        r_matrix = np.array([[np.cos(self.rela_turn), np.sin(self.rela_turn)], [-np.sin(self.rela_turn), np.cos(self.rela_turn)]])
        
        for temp_score in detect_res_pos_dict:
            for temp_rela_pos in detect_res_pos_dict[temp_score]:
                tx, ty = temp_rela_pos[0], temp_rela_pos[1]                
                center_loc_in_ref = np.dot(r_matrix, np.array([ty,tx])) + rela_loc

                new_intention = Node(node_type="intention_node", rela_cx=center_loc_in_ref[0], rela_cy=center_loc_in_ref[1], parent_node=self.current_node, score=temp_score)
                self.current_node.sub_intentions.append(new_intention)
                self.intention_nodes.append(new_intention)
                self.all_nodes.append(new_intention)

    def add_request_feature(self, rgb_image_ls, object_text):
        answer_ls = request_llm(rgb_image_ls, object_text)
        room_flag, object_flag = answer_ls[0], answer_ls[1]
        self.current_node.room_flag = room_flag
        self.current_node.object_flag = object_flag
        
    
    
    def update(self, rgb_image_ls, depth, object_text):
        point_for_close_loop_detection, laser_2d_filtered, laser_2d_filtered_angle = \
        get_laser_point(depth)
        # 表示最新的laser感知信息
        self.point_for_close_loop_detection = point_for_close_loop_detection
        self.laser_2d_filtered = laser_2d_filtered
        self.laser_2d_filtered_angle = laser_2d_filtered_angle
        
        flag, predict_node, [final_theta, final_t], [theta_to_current, t_to_current], ratio = \
        find_current_node(self.explored_nodes, self.current_node, point_for_close_loop_detection, self.rela_turn, np.array([self.rela_cx, self.rela_cy]))


        # update explored node
        if flag == True:
            last_node = self.current_node
            world_cx, world_cy, world_cz, world_turn = get_current_world_pos(self.habitat_env)
            if(last_node is None):
                self.set_current_pos(0.0, 0.0, 0.0)
                predict_node = Node(node_type="explored_node", world_cx=world_cx, world_cy=world_cy, world_cz=world_cz, world_turn=world_turn, pc=point_for_close_loop_detection)
                predict_node.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, np.array([0,0]), 0.0)
                self.current_node = predict_node
                self.all_nodes.append(self.current_node)
                self.explored_nodes.append(self.current_node)
            else:
                self.set_current_pos(0.0, 0.0, 0.0)
                predict_node = Node(node_type="explored_node", world_cx=world_cx, world_cy=world_cy, world_cz=world_cz, world_turn=world_turn, pc=point_for_close_loop_detection)
                predict_node.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, np.array([0,0]), 0.0)
                self.current_node = predict_node
                
                last_node.add_neighbor(self.current_node, t_to_current, theta_to_current)
                R = np.array([[np.cos(-theta_to_current), np.sin(-theta_to_current)], [-np.sin(-theta_to_current), np.cos(-theta_to_current)]])
                inverse_theta = -theta_to_current
                inverse_t = np.dot(R, -t_to_current)
                self.current_node.add_neighbor(last_node, inverse_t, inverse_theta)
                
                self.all_nodes.append(self.current_node)
                self.explored_nodes.append(self.current_node)
            

        elif flag == False and predict_node.name != self.current_node.name:
            self.set_current_pos(final_t[0], final_t[1], final_theta)

            last_node = self.current_node
            self.current_node = predict_node

            R1 = np.array([[np.cos(-theta_to_current), np.sin(-theta_to_current)], [-np.sin(-theta_to_current), np.cos(-theta_to_current)]])
            theta_last_in_current_p = -theta_to_current
            t_last_in_current_p = np.dot(R1, -t_to_current)
            R2 = np.array([[np.cos(-final_theta), np.sin(-final_theta)], [-np.sin(-final_theta), np.cos(-final_theta)]])
            theta_predicted_in_current_p = -final_theta
            t_predicted_in_current_p = np.dot(R2, -final_t)

            predicted_t_in_last = get_absolute_pos(t_predicted_in_current_p, t_to_current, theta_to_current)
            predicted_theta_in_last = theta_predicted_in_current_p + theta_to_current
            last_node.add_neighbor(self.current_node, predicted_t_in_last, predicted_theta_in_last)
            
            last_t_in_predicted = get_absolute_pos(t_last_in_current_p, final_t, final_theta)
            last_theta_in_predicted = theta_last_in_current_p + final_theta
            self.current_node.add_neighbor(last_node, last_t_in_predicted, last_theta_in_predicted)            
            self.current_node.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, final_t, final_theta)


        else:
            self.set_current_pos(final_t[0], final_t[1], final_theta)
            self.current_node.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, final_t, final_theta)


        # update frontier node
        candidate_frontier_arr = predict_frontier(args.init_predict_ghost_thre1, laser_2d_filtered, laser_2d_filtered_angle)
        res_frontier_pos_arr = self.multi_check_frontier(candidate_frontier_arr)
        final_frontier_pos_arr = self.select_see_ghost(res_frontier_pos_arr)
        self.add_ghost(final_frontier_pos_arr)
        self.frontier_delete()

        # update intention node
        detect_res_pos_dict = object_detect(rgb_image_ls, depth, object_text)
        
        self.add_intention(detect_res_pos_dict)

        # request the llm for question
        if(flag==True):
            s_llm_time = time.time()
            self.add_request_feature(rgb_image_ls, object_text)
            e_llm_time = time.time()
            print("=====> delta_llm_time <=====", e_llm_time-s_llm_time)
    
    # 在决策之前，先判断action_pace是否为空？若为空，才进行该操作
    def ghost_patch(self):
        predict_ghost_thre1 = args.init_predict_ghost_thre1
        while predict_ghost_thre1>=0:
            candidate_frontier_arr = predict_frontier(predict_ghost_thre1, self.laser_2d_filtered, self.laser_2d_filtered_angle)
            res_frontier_pos_arr = self.multi_check_frontier(candidate_frontier_arr)
            final_frontier_pos_arr = self.select_see_ghost(res_frontier_pos_arr)
            if(final_frontier_pos_arr.shape[0]>0):
                break
            else:
                predict_ghost_thre1 -= 0.1
        self.add_ghost(final_frontier_pos_arr)
        self.frontier_delete()

        


        


            
        

    
        
        


        


        

        