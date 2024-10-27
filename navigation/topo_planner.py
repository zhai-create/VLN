
import copy
import numpy as np

from graph.tools import find_node_path, get_absolute_pos
from perception.arguments import args as perception_args
from graph.arguments import args as graph_args

from navigation.tools import is_temp_node_see

half_len = (int)(perception_args.depth_scale/graph_args.resolution)

class TopoPlanner(object):
    """
        Plan on the topo-graph
    """
    def __init__(self, topo_graph, action_node):
        """
            Attributes
            ----------
            prior_node_ls: Nodes involved in stitching.
            topo_graph: Current topo-graph.
            action_node: Selected action node based on RL policy.
            rela_object_cx: Action node rela_cx.
            rela_object_cy: Action node rela_cy.
            object_parent_node: The parent node of the action node.
            sub_map_node: The parent node to which the sub map used for local navigation belongs.
            state_flag: Topo_planner state flag.
            remain_nodes: Remaining no-navigated nodes on the topo-path.
            origin_len_remain: Remain_nodes length before updating topo-path.
        """
        self.prior_node_ls = []        
        self.topo_graph = topo_graph
        self.action_node = action_node

        # selected sub-goal pose
        self.rela_object_cx = action_node.rela_cx
        self.rela_object_cy = action_node.rela_cy
        self.object_parent_node = action_node.parent_node
        self.sub_map_node = None
        self.state_flag = "init"
        # init: start
        # node_path: in the node path
        # finish: reach the final node in the node path

        self.remain_nodes = [] 
        self.origin_len_remain = -1
        
    def get_topo_path(self):
        """
            Get the topo node_path, node list for map stitching, remain node list
        """
        self.prior_node_ls = []
        if(self.state_flag=="init"):
            self.remain_nodes = find_node_path(self.topo_graph.current_node, self.object_parent_node, self.topo_graph.explored_nodes)
        
        temp_node_path_index = 0
        self.origin_len_remain = len(self.remain_nodes)
        while temp_node_path_index < self.origin_len_remain:  
            if(temp_node_path_index==0):
                self.prior_node_ls.append(self.remain_nodes[temp_node_path_index])
            else:
                # 判断remain_nodes[temp_node_path_index]对应的中心在remain_nodes[0]中是否可见
                if(is_temp_node_see(self.remain_nodes[0], self.remain_nodes[temp_node_path_index], 0, 0, 0)==True):
                    self.prior_node_ls.append(self.remain_nodes[temp_node_path_index])
                else:
                    break
            temp_node_path_index += 1
        self.remain_nodes = self.remain_nodes[temp_node_path_index-1:] # 一定会留下frontier或label goal所属的结点

    def sub_map_stitching(self):
        """
            Sub map stitching
            :return obstacle_map: stitching map based on the nodes in the prior_node_ls
        """
        obstacle_map = copy.deepcopy(self.sub_map_node.occupancy_map[:,:,:])
        # 地图拼接开始(将prior_node的栅格地图转换到sub_map_node的栅格地图上)
        # 遍历上一张图的所有栅格点，并将对应到当前图在200*200范围内 并且 对应到当前图中位置为0.5的栅格全部改变其数值
        for temp_prior_node in self.prior_node_ls:
            if(temp_prior_node.name != self.sub_map_node.name):
                prior_node_in_end_node = self.sub_map_node.all_other_nodes_loc[temp_prior_node.name] # prior_node在end_node坐标系下的位置和角度
                # =====================================
                origin_row_col_indices = np.argwhere(temp_prior_node.occupancy_map[:, :, 0] != graph_args.unknown_val)
                for temp_row_col_index in range(origin_row_col_indices.shape[0]):
                    row_index, column_index = origin_row_col_indices[temp_row_col_index][0], origin_row_col_indices[temp_row_col_index][1] 

                    rela_row_column_loc = get_absolute_pos(np.array([perception_args.depth_scale-graph_args.resolution*row_index, graph_args.resolution*column_index-perception_args.depth_scale]), prior_node_in_end_node[:2], prior_node_in_end_node[2])
                    row_column_t2 = rela_row_column_loc/graph_args.resolution
                    row_column_p2 = np.array([-row_column_t2[0], row_column_t2[1]])
                    row_column_loc = row_column_p2+np.array([half_len, half_len])

                    new_row_index = round(row_column_loc[0])
                    new_column_index = round(row_column_loc[1])
                    if(new_row_index>=0 and new_row_index<self.sub_map_node.occupancy_map.shape[0] and new_column_index>=0 and new_column_index<self.sub_map_node.occupancy_map.shape[1] and temp_prior_node.occupancy_map[row_index, column_index, 0]!=graph_args.unknown_val and self.sub_map_node.occupancy_map[new_row_index, new_column_index, 0]==graph_args.unknown_val):
                        obstacle_map[new_row_index, new_column_index, 0] = temp_prior_node.occupancy_map[row_index, column_index, 0]
                # =====================================
            # 地图拼接结束
        return obstacle_map
    
    def get_start_end_point(self):
        """
            Get the start point, end point, stitching map for local planning
            :return start point: start pos for local planning
            :return end point: end pos for local planning
            :return stitching map: Map used for local planning
        """
        self.get_topo_path()
        self.sub_map_node = self.prior_node_ls[0]
        if(self.origin_len_remain==1):
            self.state_flag = "finish"
        else:
            self.state_flag = "node_path"

        stitching_map = self.sub_map_stitching()
        
        # get the robot loc in the sub_map
        # ===================> get_the_start_point <===================
        if self.sub_map_node.name == self.topo_graph.current_node.name:
            rela_loc = np.array([self.topo_graph.rela_cx, self.topo_graph.rela_cy])
            rela_theta = self.topo_graph.rela_turn
        else:
            current_node_in_end_node = self.sub_map_node.all_other_nodes_loc[self.topo_graph.current_node.name] # 当前node在rrt_node坐标系下的位置和角度
            rela_loc = get_absolute_pos(np.array([self.topo_graph.rela_cx, self.topo_graph.rela_cy]), current_node_in_end_node[:2], current_node_in_end_node[2])
            rela_theta = self.topo_graph.rela_turn + current_node_in_end_node[2]

        t1 = rela_loc / graph_args.resolution 
        p1 = np.array([-t1[0], t1[1]])
        start_point = p1+np.array([half_len, half_len]) # 单位：格，有小数，数组坐标。以current_rrt_node为参考系
        # =============================================================

        # ===================> get_the_end_point <===================
        if(self.state_flag=="node_path"):
            if(self.prior_node_ls[0].name==self.prior_node_ls[-1].name):
                p2 = np.array([0, 0]) 
                end_point = p2+np.array([half_len, half_len])
            else:
                far_node_in_near_node = self.prior_node_ls[0].all_other_nodes_loc[self.prior_node_ls[-1].name]
                rela_end_loc = get_absolute_pos(np.array([0, 0]), far_node_in_near_node[:2], far_node_in_near_node[2])
                t2 = rela_end_loc/graph_args.resolution
                p2 = np.array([-t2[0], t2[1]])
                end_point = p2+np.array([half_len, half_len])
        else:
            if(self.prior_node_ls[0].name==self.prior_node_ls[-1].name):
                rela_label_loc = np.array([self.rela_object_cx, self.rela_object_cy])
                t2 = rela_label_loc/graph_args.resolution
                p2 = np.array([-t2[0], t2[1]])
                end_point = p2+np.array([half_len, half_len])
            else:
                far_node_in_near_node = self.prior_node_ls[0].all_other_nodes_loc[self.prior_node_ls[-1].name]
                new_rela_label_loc = get_absolute_pos(np.array([self.rela_object_cx, self.rela_object_cy]), far_node_in_near_node[:2], far_node_in_near_node[2])
                t2 = new_rela_label_loc/graph_args.resolution
                p2 = np.array([-t2[0], t2[1]])
                end_point = p2+np.array([half_len, half_len])
        # =============================================================
        return start_point, end_point, stitching_map


