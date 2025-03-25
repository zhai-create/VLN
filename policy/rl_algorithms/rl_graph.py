
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.cluster import DBSCAN

from graph.arguments import args as graph_args
from perception.arguments import args as perception_args
from policy.rl_algorithms.arguments import args

from env_tools.arguments import args as env_args

from graph.tools import get_absolute_pos
from collections import Counter
from navigation.tools import get_relative_pos_world
from graph.node_utils import Node

from navigation.habitat_action import HabitatAction


half_len = (int)(perception_args.graid_map_scale/graph_args.resolution)
dbscan = DBSCAN(eps=1.5, min_samples=1)
dbscan_intention = DBSCAN(eps=0.5, min_samples=1)

class RL_Graph(object):
    def __init__(self):
        self.data = {}
        self.now_node_index = 0 # 在rl_graph中当前node的index
        self.all_nodes = []
        self.all_action_nodes = []


    def reset(self):
        self.data = {}
        self.now_node_index = 0 
        self.all_nodes = []
        self.all_action_nodes = []

        self.data.update({"arrive": False}) 
        self.data.update({"state": {}})
        self.data.update({"reward": 0})
        self.data['state'].update({"current_idx": torch.Tensor([[0]])}) 
        temp_index_ls = [0 for i in range(args.graph_num_action_padding)]
        self.data['state'].update({"action_idxes": torch.Tensor([temp_index_ls])})
        temp_mask_ls = [0 for i in range(args.graph_num_action_padding)]
        self.data['state'].update({"action_mask": torch.Tensor([temp_mask_ls])})

        node_attr_ls = torch.Tensor([]) # 存储node特征
        edge_index_ls = torch.Tensor([[], []]) # 存储边的结构信息
        edge_attr_ls = torch.Tensor([]) # 存储edge特征
        new_pyg_graph = Data(x=node_attr_ls, edge_index=edge_index_ls, edge_attr=edge_attr_ls)
        self.data['state'].update({"pyg_graph": new_pyg_graph}) 


    
    def is_object_see(self, temp_intention_node):
        temp_parent_node = temp_intention_node.parent_node
        temp_parent_node_obstacle_map = temp_parent_node.occupancy_map[:,:,0]
    
        object_rela_cx, object_rela_cy = temp_intention_node.rela_cx, temp_intention_node.rela_cy
        object_rela_loc = np.array([object_rela_cx, object_rela_cy])
        object_t2 = object_rela_loc/graph_args.resolution
        object_p2 = np.array([-object_t2[0], object_t2[1]])
        end = object_p2+np.array([half_len, half_len])


        if(int(end[0])>=0 and int(end[0])<temp_parent_node_obstacle_map.shape[0] and  int(end[1])>=0 and int(end[1])<temp_parent_node_obstacle_map.shape[1] and temp_parent_node_obstacle_map[int(end[0])][int(end[1])]<graph_args.unknown_val):
            return True

        if(int(end[0])>=0 and int(end[0])<half_len and int(end[1])>=0 and int(end[1])<half_len):
            lower_bound_x = max(0, int(end[0]))
            lower_bound_y = max(0, int(end[1]))                                            
            upper_bound_x = min(temp_parent_node_obstacle_map.shape[0]-1, int(end[0])+args.is_see_grid_delta)
            upper_bound_y = min(temp_parent_node_obstacle_map.shape[1]-1, int(end[1])+args.is_see_grid_delta)
        elif (int(end[0])>=0 and int(end[0])<half_len and int(end[1])>=half_len and int(end[1])<2*half_len):
            lower_bound_x = max(0, int(end[0]))
            lower_bound_y = max(0, int(end[1])-args.is_see_grid_delta)
            upper_bound_x = min(temp_parent_node_obstacle_map.shape[0]-1, int(end[0])+args.is_see_grid_delta)
            upper_bound_y = min(temp_parent_node_obstacle_map.shape[1]-1, int(end[1]))
        elif (int(end[0])>=half_len and int(end[0])<2*half_len and int(end[1])>=0 and int(end[1])<half_len):
            lower_bound_x = max(0, int(end[0])-args.is_see_grid_delta)
            lower_bound_y = max(0, int(end[1]))
            upper_bound_x = min(temp_parent_node_obstacle_map.shape[0]-1, int(end[0]))
            upper_bound_y = min(temp_parent_node_obstacle_map.shape[1]-1, int(end[1])+args.is_see_grid_delta)
        else:
            lower_bound_x = max(0, int(end[0])-args.is_see_grid_delta)
            lower_bound_y = max(0, int(end[1])-args.is_see_grid_delta)
            upper_bound_x = min(temp_parent_node_obstacle_map.shape[0]-1, int(end[0]))
            upper_bound_y = min(temp_parent_node_obstacle_map.shape[1]-1, int(end[1]))

        for grid_x in range(lower_bound_x, upper_bound_x+1):
            for grid_y in range(lower_bound_y, upper_bound_y+1):
                if(temp_parent_node_obstacle_map[grid_x][grid_y]<graph_args.unknown_val):                    
                    return True
        return False

    
    def select_intention(self, topo_graph):
        object_score_ls = []
        for temp_intention_node in topo_graph.intention_nodes:
            if(temp_intention_node.is_see==True):
                object_score_ls.append(temp_intention_node)
            else:
                if(self.is_object_see(temp_intention_node)==True):
                    temp_intention_node.is_see = True
                    object_score_ls.append(temp_intention_node)   
                else:
                    temp_intention_node.is_see = False

        # object_score_ls = sorted(object_score_ls, key=lambda node: node.score, reverse=True)
        # if(len(object_score_ls)>args.score_top_k):
        #     object_score_ls = object_score_ls[0:args.score_top_k]
        # else:
        #     object_score_ls = object_score_ls[:]
        return object_score_ls
    
    def get_clustered_intention(self, selected_intention_node_ls):
        all_intention_world_loc = []
        for temp_index in range(len(selected_intention_node_ls)):
            temp_node = selected_intention_node_ls[temp_index]
            all_intention_world_loc.append([temp_node.world_cx, temp_node.world_cy])
        if(len(all_intention_world_loc)==0):
            return []
        all_intention_world_loc = np.array(all_intention_world_loc)
        all_intention_labels = dbscan_intention.fit_predict(all_intention_world_loc)

        cluster_res_dict = {}
        for temp_index in range(len(all_intention_labels)):
            if(all_intention_labels[temp_index] not in cluster_res_dict):
                cluster_res_dict[all_intention_labels[temp_index]] = [selected_intention_node_ls[temp_index]]
            else:
                cluster_res_dict[all_intention_labels[temp_index]].append(selected_intention_node_ls[temp_index])
        
        intention_clustered_res_ls = []
        for temp_key in cluster_res_dict:
            cluster_node_ls = cluster_res_dict[temp_key]
            is_find_center = False
            min_name_eval = 10000
            min_node = None
            for temp_node_in_cluster in cluster_node_ls:
                if(temp_node_in_cluster.name in HabitatAction.cluster_intention_name_history):
                    intention_clustered_res_ls.append(temp_node_in_cluster)
                    is_find_center = True
                    break
                else:
                    if(eval(temp_node_in_cluster.name)<min_name_eval) or (min_node is None):
                        min_name_eval = eval(temp_node_in_cluster.name)
                        min_node = temp_node_in_cluster

            if(is_find_center==False):
                intention_clustered_res_ls.append(min_node)
                HabitatAction.cluster_intention_name_history.append(min_node.name)
        return intention_clustered_res_ls    
        
    
    def get_clustered_frontier(self, topo_graph):
        all_frontier_world_loc = []
        for temp_index in range(len(topo_graph.frontier_nodes)):
            temp_node = topo_graph.frontier_nodes[temp_index]
            all_frontier_world_loc.append([temp_node.world_cx, temp_node.world_cy])
        if(len(all_frontier_world_loc)==0):
            return []
        all_frontier_world_loc = np.array(all_frontier_world_loc)
        all_frontier_labels = dbscan.fit_predict(all_frontier_world_loc)

        cluster_res_dict = {}
        for temp_index in range(len(all_frontier_labels)):
            if(all_frontier_labels[temp_index] not in cluster_res_dict):
                cluster_res_dict[all_frontier_labels[temp_index]] = [topo_graph.frontier_nodes[temp_index]]
            else:
                cluster_res_dict[all_frontier_labels[temp_index]].append(topo_graph.frontier_nodes[temp_index])

        frontier_clustered_res_ls = []
        for temp_key in cluster_res_dict:
            cluster_node_ls = cluster_res_dict[temp_key]
            
            min_dis = 10000
            min_node = None
            for temp_node_in_cluster in cluster_node_ls:
                if temp_node_in_cluster.parent_node.name != topo_graph.current_node.name:
                    n_in_current_node = topo_graph.current_node.all_other_nodes_loc[temp_node_in_cluster.parent_node.name] # 将node中的ghost坐标位置转换到当前node下
                    g_ref_loc = get_absolute_pos(np.array([temp_node_in_cluster.rela_cx, temp_node_in_cluster.rela_cy]), n_in_current_node[:2], n_in_current_node[2])
                else:
                    g_ref_loc = np.array([temp_node_in_cluster.rela_cx, temp_node_in_cluster.rela_cy])

                temp_dis = ((g_ref_loc[0]-topo_graph.rela_cx)**2+(g_ref_loc[1]-topo_graph.rela_cy)**2)**0.5
                if temp_dis<min_dis:
                    min_dis = temp_dis
                    min_node = temp_node_in_cluster
            
            frontier_clustered_res_ls.append(min_node)
        return frontier_clustered_res_ls


    def update(self, topo_graph):
        self.reset()
        selected_intention_node_ls = self.select_intention(topo_graph)
        # =======update node feature=======
        for temp_node in topo_graph.all_nodes:
            if(temp_node.node_type=="explored_node"):
                feature_ls = [-1 for i in range(80)]+[-2 for i in range(80)]+[0]
                self.data['state']['pyg_graph'].x = torch.cat([self.data['state']['pyg_graph'].x, torch.Tensor([feature_ls])], dim=0)
                temp_node.rl_node_index = len(self.all_nodes)
                self.all_nodes.append(temp_node)
            # elif(temp_node.node_type=="frontier_node"):
            #     if(len(self.all_action_nodes)>=args.graph_num_action_padding):
            #         continue
            #     self.data['state']['pyg_graph'].x = torch.cat([self.data['state']['pyg_graph'].x, torch.Tensor([[0, 0.5]])], dim=0)
            #     temp_node.rl_node_index = len(self.all_nodes)
            #     self.all_nodes.append(temp_node)

            #     action_ls_index = len(self.all_action_nodes)
            #     self.data['state']['action_idxes'][0][action_ls_index] = len(self.all_nodes)-1 # 记录在当前result['state']['pyg_graph'].x中的index位置
            #     self.data['state']['action_mask'][0][action_ls_index] = 1.0
            #     self.all_action_nodes.append(temp_node)
            # elif(temp_node.node_type=="intention_node" and ((temp_node in selected_intention_node_ls))):
            #     if(len(self.all_action_nodes)>=args.graph_num_action_padding):
            #         continue
                
            #     feature_ls = [temp_node.score_ls[i] for i in range(80)]+[temp_node.dis_ls[i] for i in range(80)]+[temp_node.intention_type]
            #     self.data['state']['pyg_graph'].x = torch.cat([self.data['state']['pyg_graph'].x, torch.Tensor([feature_ls])], dim=0)
            #     temp_node.rl_node_index = len(self.all_nodes)
            #     self.all_nodes.append(temp_node)
                
            #     action_ls_index = len(self.all_action_nodes)
            #     self.data['state']['action_idxes'][0][action_ls_index] = len(self.all_nodes)-1
            #     self.data['state']['action_mask'][0][action_ls_index] = 1.0
            #     self.all_action_nodes.append(temp_node)


        frontier_clustered_res_ls = self.get_clustered_frontier(topo_graph)
        for temp_node in frontier_clustered_res_ls:
            if(len(self.all_action_nodes)>=args.graph_num_action_padding):
                continue
            feature_ls = [-1 for i in range(80)]+[-2 for i in range(80)]+[0.5]
            self.data['state']['pyg_graph'].x = torch.cat([self.data['state']['pyg_graph'].x, torch.Tensor([feature_ls])], dim=0)
            temp_node.rl_node_index = len(self.all_nodes)
            self.all_nodes.append(temp_node)

            action_ls_index = len(self.all_action_nodes)
            self.data['state']['action_idxes'][0][action_ls_index] = len(self.all_nodes)-1 # 记录在当前result['state']['pyg_graph'].x中的index位置
            self.data['state']['action_mask'][0][action_ls_index] = 1.0
            self.all_action_nodes.append(temp_node)
                

        intention_clustered_res_ls = self.get_clustered_intention(selected_intention_node_ls)
        for temp_node in intention_clustered_res_ls:
            if(len(self.all_action_nodes)>=args.graph_num_action_padding):
                continue
            feature_ls = [temp_node.score_ls[i] for i in range(80)]+[temp_node.dis_ls[i] for i in range(80)]+[temp_node.intention_type]
            self.data['state']['pyg_graph'].x = torch.cat([self.data['state']['pyg_graph'].x, torch.Tensor([feature_ls])], dim=0)
            temp_node.rl_node_index = len(self.all_nodes)
            self.all_nodes.append(temp_node)

            action_ls_index = len(self.all_action_nodes)
            self.data['state']['action_idxes'][0][action_ls_index] = len(self.all_nodes)-1
            self.data['state']['action_mask'][0][action_ls_index] = 1.0
            self.all_action_nodes.append(temp_node)

        
        for temp_node in self.all_nodes:
            if(temp_node.node_type!="explored_node"):
                self.data['state']['pyg_graph'].edge_attr = torch.cat([self.data['state']['pyg_graph'].edge_attr, torch.Tensor([[(temp_node.rela_cx**2+temp_node.rela_cy**2)**0.5, np.arctan2(temp_node.rela_cy, temp_node.rela_cx), 0]])], dim=0)
                self.data['state']['pyg_graph'].edge_attr = torch.cat([self.data['state']['pyg_graph'].edge_attr, torch.Tensor([[(temp_node.rela_cx**2+temp_node.rela_cy**2)**0.5, np.arctan2((-1)*temp_node.rela_cy, (-1)*temp_node.rela_cx), 0]])], dim=0)
                self.data['state']['pyg_graph'].edge_index = torch.cat([self.data['state']['pyg_graph'].edge_index, torch.Tensor([[temp_node.parent_node.rl_node_index], [temp_node.rl_node_index]])], dim=1)
                self.data['state']['pyg_graph'].edge_index = torch.cat([self.data['state']['pyg_graph'].edge_index, torch.Tensor([[temp_node.rl_node_index], [temp_node.parent_node.rl_node_index]])], dim=1)
            else:
                for temp_neighbor_name in temp_node.neighbor:
                    self.data['state']['pyg_graph'].edge_attr = torch.cat([self.data['state']['pyg_graph'].edge_attr, torch.Tensor([[(temp_node.all_other_nodes_loc[temp_neighbor_name][0]**2+temp_node.all_other_nodes_loc[temp_neighbor_name][1]**2)**0.5, np.arctan2(temp_node.all_other_nodes_loc[temp_neighbor_name][1], temp_node.all_other_nodes_loc[temp_neighbor_name][0]), temp_node.all_other_nodes_loc[temp_neighbor_name][2]]])], dim=0)
                    self.data['state']['pyg_graph'].edge_index = torch.cat([self.data['state']['pyg_graph'].edge_index, torch.Tensor([[temp_node.rl_node_index], [topo_graph.get_node(temp_neighbor_name).rl_node_index]])], dim=1)

        self.now_node_index = topo_graph.current_node.rl_node_index
        self.data['state'].update({"current_idx": torch.Tensor([[self.now_node_index]])})
        self.data['state']['current_idx'] = torch.tensor(self.data['state']['current_idx'], dtype=torch.int64)
        
        self.data['state']['pyg_graph'].edge_index = torch.tensor(self.data['state']['pyg_graph'].edge_index, dtype=torch.int64)
        self.data['state']['action_idxes'] = torch.tensor(self.data['state']['action_idxes'], dtype=torch.int64)