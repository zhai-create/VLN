import numpy as np
from graph.tools import fix_size, inverse_scanner, get_absolute_pos, find_node_path
from perception.arguments import args as perception_args
from graph.arguments import args



half_len = (int)(perception_args.depth_scale/args.resolution)

class Node(object):

    name_val = 0

    def __init__(self, node_type, rela_cx=0, rela_cy=0, world_cx=None, world_cy=None, world_cz=None, world_turn=None, parent_node=None, score=0.0, pc=None):
        self.node_type = node_type # 直接赋值, str
        self.name = str(Node.name_val) # 在graph update时赋值, str
        Node.name_val += 1

        self.rela_cx = rela_cx # explored用默认值，其他结点需要赋值, float
        self.rela_cy = rela_cy # frontiet的cx和cy分别为ghost.middle[0]和middle[1], float

        self.world_cx = world_cx
        self.world_cy = world_cy
        self.world_cz = world_cz
        self.world_turn = world_turn


        self.dis = (rela_cx**2+rela_cy**2)**0.5 # float

        self.is_see = False # bool

        self.rl_node_index = -1 # int
        
        self.deleted_frontiers = [] # rl_step中实际行走步数为0的frontier

        if(node_type=="explored_node"):
            self.rela_angle_parent_center = 0
            self.rela_angle_current_center = 0
            self.occupancy_map = np.ones((2*half_len+1,2*half_len+1,1), dtype=np.float16)/2
        else:
            self.rela_angle_parent_center = np.arctan2(rela_cy, rela_cx)
            self.rela_angle_current_center = np.arctan2((-1)*rela_cy, (-1)*rela_cx)
            self.occupancy_map = None

        self.score = score # float, 只有intention需要，其他两种node均为0
        self.parent_node = parent_node

        self.all_other_nodes_loc = {} # dict
        self.neighbor = []
        self.sub_frontiers = []
        self.sub_intentions = []

        self.room_flag = -1
        self.receptacle_flag = -1
        self.guide_flag = -1
        self.object_flag = -1

        self.pc = pc

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        return False

    def __hash__(self):
        # 返回self.value的哈希值
        return hash(self.name)
    
    
    def add_neighbor(self, neighbor, relative_loc, relative_turn): # 添加当前结点的邻居结点，及其相对于当前结点的位置和角度
        if neighbor.name not in self.neighbor:
            self.neighbor.append(neighbor.name)
            self.all_other_nodes_loc.update({neighbor.name: np.array([relative_loc[0], relative_loc[1], relative_turn])})

    def add_other_node(self, other, explored_nodes): # 添加当前结点通路上除了邻居结点以外的其他结点，及其相对于当前结点的位置和角度
        if other.name not in self.all_other_nodes_loc:
            path = find_node_path(self, other, explored_nodes)
            p = path[-2]
            [front1, right1, relative_turn1] = p.all_other_nodes_loc[other.name] # 当前结点邻居与其邻居的邻居的关系
            if p.name not in self.all_other_nodes_loc:
                self.add_other_node(p, explored_nodes) 
            [front2, right2, relative_turn2] = self.all_other_nodes_loc[p.name] # 当前结点与其邻居的关系
            [front3, right3] = get_absolute_pos(np.array([front1, right1]), np.array([front2, right2]), relative_turn2) # 将path通路上所有结点的位置都转换到当前结点坐标系下
            relative_turn3 = relative_turn1 + relative_turn2
            self.all_other_nodes_loc.update({other.name: np.array([front3, right3, relative_turn3])})
    
    
    def update_occupancy(self, laser_2d_filtered, laser_2d_filtered_angle, relative_loc, relative_turn):
        laser_2d_filtered, laser_2d_filtered_angle = fix_size(laser_2d_filtered, laser_2d_filtered_angle)
        sub_map = inverse_scanner(laser_2d_filtered, laser_2d_filtered_angle, relative_loc, relative_turn)

        temp_map = np.subtract(1.0, self.occupancy_map)
        temp_map = np.divide(self.occupancy_map, temp_map)
        temp_map = np.log(temp_map)
        temp_map_new = np.subtract(1.0, sub_map)
        temp_map_new = np.divide(sub_map, temp_map_new)
        temp_map_new = np.log(temp_map_new)
        temp_map = temp_map + temp_map_new
        del temp_map_new
        temp_map = np.clip(temp_map, args.clip_lower, args.clip_upper)

        temp_map = np.exp(temp_map)
        temp_map = np.add(1.0, temp_map)
        updated_map = np.divide(1.0, temp_map)
        del temp_map
        del self.occupancy_map
        om = np.subtract(1.0, updated_map)
        om = om.astype(np.float16)
        self.occupancy_map = om