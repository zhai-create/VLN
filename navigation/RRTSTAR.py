from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import cv2
from scipy.ndimage import distance_transform_edt
from numba import jit
from termcolor import cprint
import copy
import os

from navigation.arguments import args
from navigation.tools import plot_map



class RRTNode:
    """
        Node for RRT
    """
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.name = name

class RRTStar:
    """
        RRT STAR Planner
    """
    def __init__(self, obstacles, start, goal, inflation_distance=0, start_flag=True, current_node=0, current_end_node=0, max_iter=args.max_iter, min_step_size=args.min_step_size, max_step_size=args.max_step_size, goal_step_size=args.goal_step_size, goal_threshold=args.goal_threshold, goal_sample_rate=args.goal_sample_rate, radius=args.radius,
                 obstacle_free_step_size=args.obstacle_free_step_size, obstable_distance=args.obstable_distance, obstable_threshold_unout=args.obstable_threshold_unout, obstable_threshold_unin=args.obstable_threshold_unin):
        self.inflation_distance = inflation_distance
        self.obstacles = copy.deepcopy(obstacles[:,:,0])
        self.obstacles = self.inflation(self.obstacles)
        self.obstacles = self.get_free_obstacle_map(self.obstacles, start, delta=args.free_map_delta)
        self.start = RRTNode(start[0], start[1], 0)
        self.goal = RRTNode(goal[0], goal[1], -1)
        self.x_range = [0, obstacles.shape[0]]
        self.y_range = [0, obstacles.shape[1]]
        self.max_iter = max_iter
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.goal_step_size = goal_step_size
        self.goal_threshold = goal_threshold
        self.goal_sample_rate = goal_sample_rate
        self.radius = radius
        self.obstacle_free_step_size = obstacle_free_step_size
        self.obstable_threshold_unout = obstable_threshold_unout
        self.obstable_threshold_unin = obstable_threshold_unin
        self.obstable_distance = obstable_distance
        self.node_kdtree = scipy.spatial.KDTree([(start[0], start[1])])
        obstacle_points = np.array(np.where(self.obstacles > self.obstable_threshold_unout)).T
        self.obstacle_kd_tree = scipy.spatial.KDTree(obstacle_points)

        self.bool_obstacle = np.full_like(self.obstacles, False, dtype=np.bool8)
        self.bool_obstacle[self.obstacles<=self.obstable_threshold_unin] = True
        self.indices = np.transpose(np.where(self.bool_obstacle))
        self.nodes = []


    def get_free_obstacle_map(self, obstacles, position, delta):
        lower_bound_x = max(0, int(position[0])-delta)
        lower_bound_y = max(0, int(position[1])-delta)
        upper_bound_x = min(obstacles.shape[0]-1, int(position[0])+delta)
        upper_bound_y = min(obstacles.shape[1]-1, int(position[1])+delta)
        
        res_obstacles = copy.deepcopy(obstacles)
        for i in range(lower_bound_x, upper_bound_x+1):
            for j in range(lower_bound_y, upper_bound_y+1):
                if(int(res_obstacles[i, j])==1):
                    continue
                else:
                    res_obstacles[i, j] = 0.1
        return res_obstacles
    
    
    
    def planning(self):
        count = 0
        self.nodes = [self.start]

        for i in range(self.max_iter):
            if np.random.random() < self.goal_sample_rate:
                rnd = RRTNode(self.goal.x, self.goal.y, i)
            else:
                rnd_idx = self.random_sample_boolean_array(self.bool_obstacle)
                rnd = RRTNode(rnd_idx[0], rnd_idx[1], i)

            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)
            my_flag, min_dis = self.obstacle_free(nearest_node, new_node)
            nearest_node_to_new_node_dis = self.distance(nearest_node, new_node)

            if my_flag:
                count = count+1
                near_nodes = self.find_near_nodes(new_node)
                min_cost_node = nearest_node
                min_cost = self.get_cost(nearest_node) + nearest_node_to_new_node_dis
                # new_add
                new_item = 1/(min_dis+args.min_dis_delta)
                new_item *= nearest_node_to_new_node_dis
                new_item /= args.new_item_factor # gai 0.5, 0.1
                min_cost += new_item
                
                for node in near_nodes:
                    my_flag, min_dis = self.obstacle_free(node, new_node)
                    d = self.distance(node, new_node)
                    if d >= self.min_step_size:
                        cost = self.get_cost(node) + d

                        # new_add
                        new_item = 1/(min_dis+args.min_dis_delta)
                        new_item *= d
                        new_item /= args.new_item_factor # gai 0.5
                        cost += new_item

                        if cost < min_cost and my_flag: # it will judge "cost < min_cost" first
                            min_cost_node = node
                            min_cost = cost

                new_node.parent = min_cost_node
                new_node.cost = min_cost

                self.nodes.append(new_node)
                self.rewire(new_node, near_nodes)
                self.add_to_kdtree(self.nodes)

                new_node_to_goal_dis = self.distance(new_node, self.goal)
                if new_node_to_goal_dis <= self.goal_threshold:
                    final_node = RRTNode(self.goal.x, self.goal.y, -2)
                    final_node.parent = new_node
                    final_node.cost = new_node.cost + new_node_to_goal_dis
                    self.nodes.append(final_node)
                    path = self.generate_path(final_node)
                    
                    print("plan success")
                    print("RRT STEP: ", i, len(self.nodes), count)
                    print("RRT PATH LENGTH: ", len(path))
                    plot_map(self.obstacles, path, True)
                    return path
       

        print("fail")        
        print("RRT STEP: ", i, len(self.nodes), count)
        plot_map(self.obstacles, [(self.start.x, self.start.y), (self.goal.x, self.goal.y)], False)
        return None

    def inflation(self, map):
        # 计算地图中的障碍物区域
        obstacle_map = np.where(map > 0.51, 0, 1)
        # 计算距离变换图像，获取每个点到最近的障碍物的距离
        distance_transform = distance_transform_edt(obstacle_map)
        # 将距离小于阈值的点标记为障碍物区域
        inflated_obstacle_map = np.where(distance_transform <= self.inflation_distance, 1., 0.)

        a = inflated_obstacle_map<0.01
        b = map>0.49
        c = map<0.51
        d = a&b&c
        inflated_obstacle_map[d] = 0.5

        return inflated_obstacle_map

    def random_sample_boolean_array(self, A):        
        random_index = np.random.choice(len(self.indices))
        return self.indices[random_index]

    def get_nearest_node(self, node):
        distances = np.array([self.distance(node, n) for n in self.nodes])
        nearest_node = self.nodes[np.argmin(distances)]
        # _, idx = self.node_kdtree.query((node.x, node.y))
        # nearest_node = self.nodes[idx]
        return nearest_node

    def add_to_kdtree(self, node):
        self.node_kdtree = scipy.spatial.KDTree([(n.x, n.y) for n in self.nodes])

    def steer(self, from_node, to_node):
        d = self.distance(from_node, to_node)
        if d < self.min_step_size:
            step_size = self.min_step_size
        elif d > self.max_step_size:
            step_size = self.max_step_size
        else:
            step_size = d

        theta = np.arctan2(-(to_node.x - from_node.x), to_node.y - from_node.y) #正常定义下的角度值，以左下角为原点
        new_node = RRTNode(from_node.x - step_size * np.sin(theta),
                        from_node.y + step_size * np.cos(theta), to_node.name)
        return new_node

    def obstacle_free(self, from_node, to_node):
        n = round(self.distance(from_node, to_node)/self.obstacle_free_step_size)+1
        
        if n == 1:
            x_values = np.array([from_node.x, to_node.x])
            y_values = np.array([from_node.y, to_node.y])
        else:
            x_values = np.linspace(from_node.x, to_node.x, num=n)
            y_values = np.linspace(from_node.y, to_node.y, num=n)

        if from_node is self.start:
            x_values = x_values[1:]
            y_values = y_values[1:]
            
        check_points = list(zip(x_values, y_values))
        distances, _ = self.obstacle_kd_tree.query(check_points)
        min_dis = min(distances)
        flag1 = all(distances >= self.obstable_distance)
        flag2 = True
        for x, y in zip(x_values, y_values):
            try:
                if self.obstacles[round(x), round(y)] > self.obstable_threshold_unin \
                and self.obstacles[round(x), round(y)] < self.obstable_threshold_unout:
                    flag2 = False
            except:
                continue
        return flag1 and flag2, min_dis


    def find_near_nodes(self, node):
        near_nodes = [n for n in self.nodes if self.distance(n, node) <= self.radius]
        return near_nodes

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            node_to_new_node_dis = self.distance(node, new_node)
            my_flag, min_dis = self.obstacle_free(node, new_node)
            # new_add
            new_item = 1/(min_dis+args.min_dis_delta)
            new_item *= node_to_new_node_dis
            new_item /= args.new_item_factor
            if self.get_cost(new_node) + node_to_new_node_dis + new_item < self.get_cost(node):
                # if self.obstacle_free(new_node, node):
                if my_flag:
                    node.parent = new_node
                    node.cost = self.get_cost(new_node) + node_to_new_node_dis + new_item

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def get_cost(self, node):
        return node.cost

    def generate_path(self, final_node):
        path = []
        current_node = final_node
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        return path[::-1]

