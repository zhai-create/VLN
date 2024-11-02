
import math
import numpy as np
import time
import cv2
import copy
from scipy.ndimage import distance_transform_edt

from navigation.arguments import args
from navigation.tools import plot_map
from graph.arguments import args as graph_args


'''
    对象Map，主要有地图数据、起点和终点
'''
class Map(object):
    """
        Map for Astar
    """
    def __init__(self,mapdata,startx,starty,endx,endy):
        """
            Attributes
            ----------
            data: Map with starting and ending points in free areas.
            startx, starty: Start point.
            endx, endy: End poiny.
        """
        self.data = mapdata
        self.startx = startx
        self.starty = starty
        self.endx = endx
        self.endy = endy

        self.data = self.get_free_obstacle_map(self.data, np.array([startx, starty]), delta=1)
        self.data = self.get_free_obstacle_map(self.data, np.array([endx, endy]), delta=1)


    def get_free_obstacle_map(self, obstacles, position, delta):
        """
            Change the start and end points of the map to free areas.
            :param obstacles: origin map
            :param position: start point or end point.
            :param delta: free area delta.
            :return res_obstacles: Result of the free map.
        """
        res_obstacles = copy.deepcopy(obstacles)
                
        lower_bound_i = max(int(position[0])-delta, 0)
        lower_bound_j = max(int(position[1])-delta, 0)
        upper_bound_i = min(int(position[0])+delta, res_obstacles.shape[0]-1)
        upper_bound_j = min(int(position[1])+delta, res_obstacles.shape[1]-1)

        for i in range(lower_bound_i, upper_bound_i+1):
            for j in range(lower_bound_j, upper_bound_j+1):
                if(int(res_obstacles[i, j])==1):
                    continue
                else:
                    res_obstacles[i, j] = 0.1
        return res_obstacles



'''
    Node.py主要是描述对象Node
'''
class Node(object):
    """
        Node for Astar
    """
    def __init__(self,x,y,g,h,father):
        """
            Attributes
            ----------
            x, y: Node pos.
            g: Already cost.
            h: Potention cost.
            father: parent node.
        """
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.father = father
    '''
        处理边界和障碍点
    '''

    def find_nearest_grid(self, mapdata, temp_x, temp_y):
        """
            Get the nearest no-free area.
            :param mapdata: mao for astar.
            :param temp_x, temp_y: Coordinates to be investigated.
            :return min_dis: Distance to the nearest no-free area.
        """
        grid_delta = args.astar_grid_delta

        lower_bound_x = max(0, int(temp_x)-grid_delta)
        lower_bound_y = max(0, int(temp_y)-grid_delta)
        upper_bound_x = min(mapdata.shape[0]-1, int(temp_x)+grid_delta)
        upper_bound_y = min(mapdata.shape[1]-1, int(temp_y)+grid_delta)

        min_dis = -1
        min_row, min_line = -1, -1
        for temp_row in range(lower_bound_x, upper_bound_x+1):
            for temp_line in range(lower_bound_y, upper_bound_y+1):
                if(mapdata[temp_row][temp_line]>=graph_args.unknown_val):
                    temp_dis = ((temp_row-temp_x)**2+(temp_line-temp_y)**2)**0.5
                    if(temp_dis<min_dis or min_dis<0):
                        min_dis = temp_dis
                        min_row, min_line = temp_row, temp_line
        return min_dis


    def get_new_cost(self, min_dis):
        """
            Get the additional cost.
            :param min_dis: min dis to the no-free area.
            :return additional cost
        """
        if(min_dis==-1):
            return 0
        return args.cost_factor/(min_dis)

    def getNeighbor(self,mapdata,endx,endy):
        """
            Get the neighbor nodes.
            :param mapdata: map for astar.
            :return endx, endy: end point.
        """
        x = self.x
        y = self.y
        result = []
    #先判断是否在上下边界
    #if(x!=0 or x!=len(mapdata)-1):
    #上
    #Node(x,y,g,h,father)
        if(x!=0 and mapdata[x-1][y]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x-1, y)
            upNode = Node(x-1,y,self.g+10+self.get_new_cost(min_dis), np.sqrt((x-1-endx)**2+(y-endy)**2)*10,self)
            result.append(upNode)
    #下
        if(x!=len(mapdata)-1 and mapdata[x+1][y]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x+1, y)
            downNode = Node(x+1,y,self.g+10+self.get_new_cost(min_dis),np.sqrt((x+1-endx)**2+(y-endy)**2)*10,self)
            result.append(downNode)
    #左
        if(y!=0 and mapdata[x][y-1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x, y-1)
            leftNode = Node(x,y-1,self.g+10+self.get_new_cost(min_dis),np.sqrt((x-endx)**2+(y-1-endy)**2)*10,self)
            result.append(leftNode)
    #右
        if(y!=len(mapdata[0])-1 and mapdata[x][y+1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x, y+1)
            rightNode = Node(x,y+1,self.g+10+self.get_new_cost(min_dis),np.sqrt((x-endx)**2+(y+1-endy)**2)*10,self)
            result.append(rightNode)
    
    #西北  14
        if(x!=0 and y!=0 and mapdata[x-1][y-1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x-1, y-1)
            wnNode = Node(x-1,y-1,self.g+14+self.get_new_cost(min_dis),np.sqrt((x-1-endx)**2+(y-1-endy)**2)*10,self)
            result.append(wnNode)
    #东北
        if(x!=0 and y!=len(mapdata[0])-1 and mapdata[x-1][y+1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x-1, y+1)
            enNode = Node(x-1,y+1,self.g+14+self.get_new_cost(min_dis),np.sqrt((x-1-endx)**2+(y+1-endy)**2)*10,self)
            result.append(enNode)
    #西南
        if(x!=len(mapdata)-1 and y!=0 and mapdata[x+1][y-1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x+1, y-1)
            wsNode = Node(x+1,y-1,self.g+14+self.get_new_cost(min_dis),np.sqrt((x+1-endx)**2+(y-1-endy)**2)*10,self)
            result.append(wsNode)
    #东南
        if(x!=len(mapdata)-1 and y!=len(mapdata[0])-1 and mapdata[x+1][y+1]<0.5):
            min_dis = self.find_nearest_grid(mapdata, x+1, y+1)
            esNode = Node(x+1,y+1,self.g+14+self.get_new_cost(min_dis),np.sqrt((x+1-endx)**2+(y+1-endy)**2)*10,self)
            result.append(esNode)

        return result

    def hasNode(self,worklist):
        for i in worklist:
            if(i.x==self.x and i.y ==self.y):
                return True
        return False

    def changeG(self,worklist):
        for i in worklist:
            if(i.x==self.x and i.y ==self.y):
                if(i.g>self.g):
                    i.g = self.g


def getKeyforSort(element:Node):
    """
        Get the complet cost funcion value.
        :param Node: current astar node.
        :return result: complet cost funcion value.
    """
    return element.g+element.h
    # return element.g #element#不应该+element.h，否则会穿墙



def astar(workMap):
    """
        Astar planning.
        :param workMap: map for astar.
        :return result: atsar path.
    """
    try:
        init_time = time.time()
        startx,starty = workMap.startx,workMap.starty
        endx,endy = workMap.endx,workMap.endy
        startNode = Node(startx, starty, 0, 0, None)
        openList = []
        lockList = []
        lockList.append((startNode.x, startNode.y))
        currNode = startNode
        current_node_ls = []
        while((endx,endy) != (currNode.x,currNode.y)):
            workList = currNode.getNeighbor(workMap.data,endx,endy)
            end_time = time.time()
            if ((end_time-init_time)>args.time_thre):
                print("Noneeeee")
                print((end_time-init_time))
                plot_map(workMap.data, [(startx, starty), (endx, endy)], False)
                return None
            for temp_work_node in workList:
                if ((temp_work_node.x, temp_work_node.y) not in lockList): # 找到当前node附近（8格中）：没有“研究过的”&空闲区域node
                    if(temp_work_node.hasNode(openList)):
                        temp_work_node.changeG(openList)
                    else:
                        openList.append(temp_work_node)
            openList.sort(key=getKeyforSort)#关键步骤
            currNode = openList.pop(0)
            lockList.append((currNode.x, currNode.y))
        result = []
        while(currNode.father!=None):
            result.append((currNode.x,currNode.y))
            currNode = currNode.father
        result.append((currNode.x,currNode.y))
        result.reverse()
        plot_map(workMap.data, result, True)
        end_time = time.time()
        print("=====> astar_time_delta <=====", end_time-init_time)
        return result
    except:
        plot_map(workMap.data, [(startx, starty), (endx, endy)], False)
        return None
