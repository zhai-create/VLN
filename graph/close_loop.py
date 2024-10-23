#!/usr/bin/env python

# import rospy
import numpy as np
import math
from numpy import cos, sin
from scipy.spatial import cKDTree

# structure of the nearest neighbor 
# 这里的坐标系：[x右，y前，z上]，注意与habitat坐标系的不同，这里输出的t取负号等于habitat坐标系下的平移量
class NeighBor:
    def __init__(self):
        self.distances = []
        self.src_indices = []
        self.tar_indices = []

class Close_Loop:
    def __init__(self, match_tolerance=0.05, iter_tolerance=0.01, max_iter=50):
        self.match_tolerance = match_tolerance # range:0~1 referring to depth 0-10m
        self.iter_tolerance = iter_tolerance
        self.max_iter = max_iter

    # Waiting for Implementation 
    # return: T = (R, t), where T is 2*3, R is 2*2 and t is 2*1
    def process(self,tar_pc,src_pc, pre_angle, pre_t):
        # n.pc, src_pc, final_rela_turn, np.array([final_rela_t[1], final_rela_t[0], 0])
        # pre_angle:src要再转多少能到达tar的角度
        # pre_t:src转完之后，在转完后的坐标系下，平移多少能到达tar的位置
        src_pc = src_pc.T
        tar_pc = tar_pc.T
        tree = cKDTree(tar_pc)
        # src_cen = np.average(src_pc, axis=0)
        # src = src_pc - src_cen

        init_theta = pre_angle
        init_t = pre_t
        error = 50
        last_error = 100
        i = 0
        matched_ratio = 0.0
        while abs(last_error - error) / (last_error + 0.1) > self.iter_tolerance and i < self.max_iter:
            last_error = error
            transformed = self.transformPointsForward(init_theta, init_t, src_pc)
            # if i == 0:
            #     print(tar_pc, transformed)
            matched_tar, filtered_src_indices, dists, error, matched_ratio = self.findNearest(tree, transformed, tar_pc)
            # print(matched_tar)
            matched_cen = np.average(matched_tar, axis=0)
            tar = matched_tar - matched_cen
            filtered_src = src_pc[filtered_src_indices]
            filtered_src_cen = np.average(filtered_src, axis=0)
            src = filtered_src - filtered_src_cen
            # fenzi, fenmu = 0., 0.
            # for i in range(len(tar)):
            #     fenzi = fenzi + tar[i, 0] * src[i, 1] - tar[i, 1] * src[i, 0]
            #     fenmu = fenmu + tar[i, 0] * src[i, 0] + tar[i, 1] * src[i, 1]
            # init_theta = -math.atan(fenzi/fenmu)
            t2 = tar[:, 0:2]
            s2 = src[:, 0:2]
            fenzi = np.sum(np.cross(t2, s2))
            fenmu = np.sum(t2 * s2)
            # init_theta = -math.atan(fenzi / fenmu)
            init_theta = -np.arctan2(fenzi, fenmu)
            # print(init_theta/np.pi*180)
            init_t = matched_cen - np.dot(self.getR(init_theta), filtered_src_cen)
            i = i + 1
        theta, t = init_theta, init_t
        T = self.getTransform(theta, t)
        # return T
        return theta, t[:2], matched_ratio
        pass

    # find the nearest points & filter
    # return: neighbors of src and tar
    def findNearest(self,tree,src,tar):
        
        dists, indices = tree.query(src)
        valid_indices = dists < self.match_tolerance
        # print("aaaa ", dists)
        # print("valid_indices:", valid_indices)
        # print("dists[valid_indices]:", dists[valid_indices])
        error = np.sqrt(np.sum(dists[valid_indices] ** 2) / len(dists[valid_indices]))
        matched_tar = tar[indices[valid_indices]]
        matched_ratio = len(matched_tar) / len(src)

        # print(dists, indices, valid_indices, matched_ratio)
        return matched_tar, valid_indices, dists, error, matched_ratio
        # D = np.zeros(len(src))
        # I = np.zeros(len(src))
        # for i, p in enumerate(src):
        #     # print(i)
        #     ed = np.sqrt(np.sum((p - tar) ** 2, axis=1))
        #     D[i] = ed.min()
        #     I[i] = ed.argmin()
        # error = (np.sum(D ** 2) / len(D)) ** 0.5
        # Z = np.zeros([len(src), 3])
        # for i, p in enumerate(I):
        #     Z[i, :] = tar[int(p), :]
        # return Z, error
       

    # Waiting for Implementation 
    # return: T = (R, t), where T is 2*3, R is 2*2 and t is 2*1
    def getTransform(self,theta,t):
        # ...
        # ...
        T = np.array([[cos(theta), -sin(theta), t[0]], [sin(theta), cos(theta), t[1]]])
        return T
        pass

    def transformPointsForward(self,theta,t,src):
        R = self.getR(theta)
        temp = src.T
        r = np.dot(R, temp).T
        transformed = r + t
        return transformed

    def getR(self,theta):
        R = np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
        return R

if __name__ == "__main__":
    I = Close_Loop()
    src = np.array([[10,0,-10,0],[0,1,0,-1],[0,0,0,0]])
    tar = np.array([[0,-1,0,1],[10,0,-10,0],[0,0,0,0]])
    print(src)
    print(I.process(tar,src,0.1*np.pi))