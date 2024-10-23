#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import utm
import cv2
import math
import open3d
import psutil
import GPUtil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from operator import itemgetter 


#  ---------> Y  W  (image)                x (robot) 
#  |                                      A
#  |                                      |
#  v                                      |
#  x H                         y <---------
 
    
####################################
#             Functions
####################################

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

COLORS_ = np.array([[192, 192, 192],
                   [ 87, 172, 168],
                   [119, 224, 219],
                   [158, 218, 229],
                   [  0, 178, 230],
                   [ 98, 206, 236],
                   [174, 145, 230],
                   [207, 190, 240],
                   [  0,   0,   0]]) / 255.
COLORS = np.array([COLORS_[:, 2], COLORS_[:, 1], COLORS_[:, 0]]).T


LABEL_COLORS = np.array([
    (255, 255, 255), # 0 None
    (70, 70, 70),    # 1 Building
    (100, 40, 40),   # 2 Fences
    (55, 90, 80),    # 3 Other
    (220, 20, 60),   # 4 Pedestrian
    (153, 153, 153), # 5 Pole
    (157, 234, 50),  # 6 RoadLines
    (128, 64, 128),  # 7 Road
    (244, 35, 232),  # 8 Sidewalk
    (107, 142, 35),  # 9 Vegetation
    (0, 0, 142),     # 10 Vehicle
    (102, 102, 156), # 11 Wall
    (220, 220, 0),   # 12 TrafficSign
    (70, 130, 180),  # 13 Sky
    (81, 0, 81),     # 14 Ground
    (150, 100, 100), # 15 Bridge
    (230, 150, 140), # 16 RailTrack
    (180, 165, 180), # 17 GuardRail
    (250, 170, 30),  # 18 TrafficLight
    (110, 190, 160), # 19 Static
    (170, 120, 50),  # 20 Dynamic
    (45, 60, 150),   # 21 Water
    (145, 170, 100), # 22 Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def gps2xy(lat, lon):
    return utm.from_latlon(lat, lon)[:2]


def xy2gps(x, y):
    return utm.to_latlon(x, y, 51, "R")


def xy2polar(dx, dy):
    distance = math.sqrt(dx*dx+dy*dy)
    theta = pi2pi(math.atan2(dy, dx))
    return distance, theta


def vw2vxvy(v, w):
    """
    v : 0 ~ MAX_LINEAR_VEL m/s
    w : -MAX_ANGULAR_VEL ~ MAX_ANGULAR_VEL rad/s
    vx, vy : m/s (vx > 0)
    """
    vx = np.cos(w) * v
    vy = np.sin(w) * v
    return vx, vy


def vxvy2vw(vx, vy):
    v = np.sqrt(vx * vx + vy * vy)
    w = math.atan(vy / (vx + 1e-5))
    return v, w


def pi2pi(theta):
    """
    to -pi ~ pi
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi


def pi2pi_tensor(theta):
    TWO_PI = 2 * math.pi
    theta = torch.fmod(torch.fmod(theta, TWO_PI) + TWO_PI, TWO_PI)
    return torch.where(theta > math.pi, theta - TWO_PI, theta)


def calc_yaw(orientation):
    """
    from quaternion to angle in rad / deg
    """
    atan2_y = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    atan2_x = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    yaw = math.atan2(atan2_y , atan2_x)
    # print("yaw : ", np.rad2deg(yaw))    
    return yaw


def calc_euclidean_distance(x1, y1, x2, y2):
    """
    calculate the euclidean distance between two points
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calc_relative_state(ego_vx, ego_vy, ego_x, ego_y, ego_yaw, other_vx, other_vy, other_x, other_y, other_yaw, using_xy=True):
    """
    calculate the relative states of the other agents in the egocentric coordinate system
    """
    # ==================
    #      position
    # ==================
    # distance & direction or x & y
    delta_distance = calc_euclidean_distance(other_x, other_y, ego_x, ego_y)
    delta_theta = pi2pi(math.atan2(other_y - ego_y, other_x - ego_x))
    
    dx_world = other_x - ego_x
    dy_world = other_y - ego_y
    delta_dx = ( dx_world * np.cos(ego_yaw) + dy_world * np.sin(ego_yaw))
    delta_dy = (-dx_world * np.sin(ego_yaw) + dy_world * np.cos(ego_yaw))
    
    # ==================
    #      velocity
    # ==================    
    delta_yaw = other_yaw - ego_yaw
    delta_vx = ( other_vx * np.cos(delta_yaw) + other_vy * np.sin(delta_yaw)) - ego_vx
    delta_vy = (-other_vx * np.sin(delta_yaw) + other_vy * np.cos(delta_yaw)) - ego_vy

    if using_xy:
        return delta_dx, delta_dy, delta_vx, delta_vy
    else:    
        return delta_distance, delta_theta, delta_vx, delta_vy 


def cv2_rotate(img, yaw, map_range=128):
    
    # matrix
    M = cv2.getRotationMatrix2D((map_range/2, map_range/2), -yaw, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # new bound
    nW = int((map_range * sin) + (map_range * cos))
    nH = int((map_range * cos) + (map_range * sin))
    
    # translate
    M[0, 2] += (nW / 2) - map_range / 2
    M[1, 2] += (nH / 2) - map_range / 2
    
    img = cv2.warpAffine(img, M, (nW, nH))

    # center crop
    x = img.shape[0] / 2 - map_range / 2
    y = img.shape[1] / 2 - map_range / 2
    
    return img[int(x):int(x+map_range), int(y):int(y+map_range)]

            
def cv2_show_img(name, img):
        
    cv2.imshow(name, img)
    cv2.waitKey(10)


def PIL_rotate(img, yaw):
    
    img = Image.fromarray(np.array(img))
    img = np.array(img.rotate(-yaw))
    
    return img


def viz_o3d(points, colors=None):
    
    point_viz = open3d.geometry.PointCloud()
    point_viz.points = open3d.utility.Vector3dVector(points)
    
    if colors is not None:
        point_viz.colors = open3d.utility.Vector3dVector(colors)
    
    open3d.visualization.draw_geometries([point_viz])


def down_sampling(points_cloud, labels, voxel_size=1.5):
    
    down_points_cloud = open3d.geometry.PointCloud()
    down_points_cloud.points = open3d.utility.Vector3dVector(points_cloud)
    if labels is not None:
        point_color = LABEL_COLORS[labels]
        down_points_cloud.colors = open3d.utility.Vector3dVector(point_color)
    down_points_cloud = down_points_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    return np.asarray(down_points_cloud.points), np.asarray(down_points_cloud.colors)
      

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.001)
        except:
            pass
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.001)
        except:
            pass
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        nn.init.constant_(m.bias, 0.001)


def Lidar_Transformation(lidar_current, 
                         current_x, current_y, current_yaw, 
                         target_x, target_y, target_yaw,
                         laser_num, laser_range, img_h, img_w):
    index_xy = np.linspace(0, laser_num, laser_num)
    lidar_x_current = lidar_current * np.sin(index_xy / laser_num * math.pi).astype(np.float64)
    lidar_y_current = lidar_current * np.cos((1- index_xy / laser_num) * math.pi).astype(np.float64)
    ones = np.ones_like(lidar_x_current)
    lidar_coordinates_current = np.stack([lidar_x_current, lidar_y_current, ones], axis=0)

    cos_target = math.cos(target_yaw)
    sin_target = math.sin(target_yaw)
    T_target = np.array([
        [cos_target, -sin_target, target_x],
        [sin_target,  cos_target, target_y],
        [         0,           0,        1]])

    cos_current = math.cos(current_yaw)
    sin_current = math.sin(current_yaw)
    T_current = np.array([
        [cos_current, -sin_current, current_x],
        [sin_current,  cos_current, current_y],
        [          0,            0,         1]])

    T_target_inv = np.linalg.inv(T_target)
    T_current_target = T_target_inv.dot(T_current)

    lidar_coordinates_target = T_current_target.dot(lidar_coordinates_current)
    lidar_x_target = lidar_coordinates_target[0]
    lidar_y_target = lidar_coordinates_target[1]

    image_x_target = np.floor(img_h - 1.0 - lidar_x_target / laser_range * img_h).astype(np.int32)
    image_y_target = np.floor(img_h - 1.0 - lidar_y_target / laser_range * img_h).astype(np.int32)

    select_index = (image_x_target >= 0) & (image_y_target >= 0) & \
                    (image_x_target < img_h) & (image_y_target < img_w) & \
                    (lidar_current < laser_range-0.2)

    image_x = image_x_target[select_index].astype(np.int32)
    image_y = image_y_target[select_index].astype(np.int32)
    
    return image_x, image_y


def Coordinate_Transformation(d_x, d_y, current_x, current_y, current_yaw, target_x, target_y, target_yaw):
    
    T_target = np.array([
        [math.cos(target_yaw), -math.sin(target_yaw), target_x],
        [math.sin(target_yaw),  math.cos(target_yaw), target_y],                  
        [                   0,                     0,        1]])

    T_current = np.array([
        [math.cos(current_yaw), -math.sin(current_yaw), current_x], 
        [math.sin(current_yaw),  math.cos(current_yaw), current_y],
        [                    0,                      0,         1]])

    T_target_inv = np.linalg.inv(T_target)
    T_current_to_target = T_target_inv.dot(T_current)

    coordinates_current = [d_x, d_y, 1]
    coordinates_target = T_current_to_target.dot(coordinates_current)

    return coordinates_target[0], coordinates_target[1]


def CostMap(observation_list, 
            laser_dim, laser_range, img_h, img_w, obs_dim, highlight_iterations,
            highlight=True, combine=True, batch=False, black_obs=True):
    '''
    input:
        numpy.ndarray : (B, T, N) or (B, T*N) or (T, N) or (T*N)
    output:
        numpy.ndarray
    '''
    if not batch:
        observation_list = np.expand_dims(observation_list, axis=0) # (B, T, N) or (B, T*N)

    batch_size = observation_list.shape[0]
    observation_list = observation_list.reshape(batch_size, -1, obs_dim) # (B, T, N)
    length = observation_list.shape[1]

    lidar_list = observation_list[:, :, :laser_dim]
    x_list = observation_list[:, :, laser_dim]
    y_list = observation_list[:, :, laser_dim + 1]
    yaw_list = observation_list[:, :, laser_dim + 2]

    if not combine:
        if black_obs: # 0 is obstacle
            dynamic_ego_obs = np.ones((batch_size, length, img_h, img_w)) # (B, T, H, W)
        else:
            dynamic_ego_obs = np.zeros((batch_size, length, img_h, img_w)) # (B, T, H, W)
    else:
        if black_obs:
            dynamic_ego_obs = np.ones((batch_size, 1, img_h, img_w)) # (B, T, H, W)
        else:
            dynamic_ego_obs = np.zeros((batch_size, 1, img_h, img_w)) # (B, T, H, W)
    
    for m in range(batch_size):
        for i in range(length):
            image_x, image_y = Lidar_Transformation(
                lidar_list[m][i], 
                x_list[m][i], y_list[m][i], yaw_list[m][i],
                x_list[m][-1], y_list[m][-1], yaw_list[m][-1],
                laser_dim, laser_range, img_h, img_w)
            
            if not combine:
                dynamic_ego_obs[m, i, image_x, image_y] = 0 if black_obs else 1

                if highlight:
                    if black_obs:
                        dynamic_ego_obs[m, i] = cv2.erode(dynamic_ego_obs[m, i], np.ones((3,3), np.uint8), iterations=highlight_iterations)
                    else:
                        dynamic_ego_obs[m, i] = cv2.dilate(dynamic_ego_obs[m, i], np.ones((3,3), np.uint8), iterations=highlight_iterations)
            else:
                dynamic_ego_obs[m, 0, image_x, image_y] = 1 - (i+1) / length if black_obs else (i+1) / length
    
        if combine and highlight:
            if black_obs:
                dynamic_ego_obs[m, 0] = cv2.erode(dynamic_ego_obs[m, 0], np.ones((3,3), np.uint8), iterations=highlight_iterations)
            else:
                dynamic_ego_obs[m, 0] = cv2.dilate(dynamic_ego_obs[m, 0], np.ones((3,3), np.uint8), iterations=highlight_iterations)
        
    if not batch:
        dynamic_ego_obs = dynamic_ego_obs[0] # (H, W) or (T, H, W)

    return dynamic_ego_obs


def DataDisentangle(state, 
                    history_length, laser_dim, laser_range, obs_dim, vel_dim, img_h, img_w, highlight_iterations,
                    black_obs=True):
    ## Deconstruct Data
    observation = state[:obs_dim*history_length] # (T*N)
    velocity = state[obs_dim*history_length : obs_dim*history_length+vel_dim]
    
    ## Disentangle Lidar images without dilate
    disentangled_lidar_images = CostMap(observation, laser_dim, laser_range, img_h, img_w, obs_dim, highlight_iterations, highlight=False, combine=False, black_obs=black_obs) # (T, H, W)

    ## motion input
    if black_obs:
        motion_input = np.ones((history_length-2, img_h, img_w)) # (T-2, H, W)
        disentangled_lidar_images_highlight = np.ones_like(disentangled_lidar_images)
        for i in range(history_length):
            disentangled_lidar_images_highlight[i] = cv2.erode(disentangled_lidar_images[i], np.ones((3, 3), np.uint8), iterations=2)
    else:
        motion_input = np.zeros((history_length-2, img_h, img_w)) # (T-2, H, W)
        disentangled_lidar_images_highlight = np.zeros_like(disentangled_lidar_images)
        for i in range(history_length):
            disentangled_lidar_images_highlight[i] = cv2.dilate(disentangled_lidar_images[i], np.ones((3, 3), np.uint8), iterations=2)
                
    
    for i in range(history_length-2):
        before_index = i 
        after_index = i + 2

        before = disentangled_lidar_images[before_index].astype(np.uint8)
        after = disentangled_lidar_images[after_index].astype(np.uint8)

        before_highlight = disentangled_lidar_images_highlight[before_index].astype(np.uint8)
        after_highlight = disentangled_lidar_images_highlight[after_index].astype(np.uint8)
        
        if black_obs:
            delta_after_highlight = cv2.erode(
                np.clip(after_highlight.astype(np.float) - before.astype(np.float), 0, 1),
                np.ones((1, 1), np.uint8), iterations=1)
            
            delta_before_highlight = cv2.erode(
                np.clip(before_highlight.astype(np.float) - after.astype(np.float), 0, 1),
                np.ones((1, 1), np.uint8), iterations=1)
        else:
            delta_after_highlight = cv2.dilate(
                np.clip(after_highlight.astype(np.float) - before.astype(np.float), 0, 1),
                np.ones((1, 1), np.uint8), iterations=1)
            
            delta_before_highlight = cv2.dilate(
                np.clip(before_highlight.astype(np.float) - after.astype(np.float), 0, 1),
                np.ones((1, 1), np.uint8), iterations=1)

        delta_highlight = delta_after_highlight - delta_before_highlight # -1~1
        # motion_input[i] = delta_highlight
        motion_input[i] = cv2.GaussianBlur(delta_highlight, (3, 3), 0) # -1~1
    
    motion_input = torch.from_numpy(motion_input).unsqueeze(0).float() # (C, T-2, H, W)

    ## content input
    content_input = disentangled_lidar_images_highlight[-1] * 2. - 1.  # 0~1 => -1~1
    content_input = torch.from_numpy(content_input).unsqueeze(0).float() # (C, H, W)

    ## current (v, w)
    velocity_input = torch.from_numpy(velocity).float()

    return motion_input, content_input, velocity_input


def generate_heatmap_target(target, 
                            laser_range, max_distance, img_h, img_w,
                            k_target=0.4):

    RESOLUTION = laser_range / img_h

    target_distance =target[:, 0] * max_distance
    target_theta = (target[:, 1] * 2. - 1) * math.pi

    target_w = img_w / 2 -target_distance * torch.sin(target_theta) / RESOLUTION
    target_h = img_w / 2 -target_distance * torch.cos(target_theta) / RESOLUTION # 图像坐标系下

    target_positions = torch.stack([target_h, target_w], dim=1)

    batch_size = list(target_positions.shape)[0]

    target_positions_reshaped = torch.reshape(target_positions[..., :], [batch_size, 1, 1, 2])

    aranges = [torch.arange(s) for s in [img_w, img_w]]
    grid = torch.meshgrid(*aranges)
    grid_stacked = torch.stack(grid, dim=2) # (H, W, (h,w))
    grid_stacked = grid_stacked.float()
    grid_stacked = torch.stack([grid_stacked] * batch_size, dim=0).cuda() # (B, H, W, 2)

    squared_distances = torch.squeeze(torch.mean(torch.pow(grid_stacked - target_positions_reshaped, 2.0), dim=-1), dim=-1)
    heatmap = k_target * squared_distances / img_h / img_h

    heatmap = torch.clamp(heatmap[:, 0:img_h, 0:img_w], min=0.0, max=1.0)

    # Show heatmaps
    # for b in range(batch_size):
    #     print("max, min:", torch.max(heatmap[b, :, :]), torch.min(heatmap[b, :, :]))
    #     tmp = heatmap[b, :, :]
    #     tmp = tmp.cpu().detach().numpy()
    #     plt.imshow(tmp)
    #     plt.show()

    return heatmap.unsqueeze(1)


def merge_visualization(traj_img, lidar_ego_imgs,
                        num_agent, traj_img_range, img_h, img_w):
    scale = 1
    # Merge lidar img
    bar_height = int(30 * scale)
    bound_height = int((traj_img_range - bar_height * (num_agent - 1) - img_h * num_agent) / 2 * scale)
    bar = cv2.cvtColor(np.zeros((bar_height, img_w * scale)).astype(np.float32), cv2.COLOR_GRAY2RGB)
    bound = cv2.cvtColor(np.zeros((bound_height, img_w * scale)).astype(np.float32), cv2.COLOR_GRAY2RGB)

    lidar_img = cv2.vconcat([bound,
                             cv2.resize(lidar_ego_imgs[0], (img_w * scale, img_h * scale))])
    
    for i in range(num_agent - 1):
        lidar_img = cv2.vconcat([lidar_img,
                                 bar,
                                 cv2.resize(lidar_ego_imgs[i+1], (img_w * scale, img_h * scale))])
    lidar_img = cv2.vconcat([lidar_img, bound])
    
    # Merge traj img
    v_bar_width = int(20 * scale)
    v_bar = cv2.cvtColor(np.zeros((traj_img_range * scale, v_bar_width)).astype(np.float32), cv2.COLOR_GRAY2RGB)
    merge_img = cv2.hconcat([v_bar,
                             cv2.resize(traj_img, (traj_img_range * scale, traj_img_range * scale)).astype(np.float32), 
                             v_bar,
                             lidar_img,
                             v_bar])
    
    return merge_img


def system_info(init_process_memory, init_free_memory, if_bar=False):
    
    process_info = psutil.Process(os.getpid()).memory_full_info()
    system_memory_info = psutil.virtual_memory()
    gpu = GPUtil.getGPUs()[0]
    
    scale =  1. / 1024. / 1024. / 1024.
    print("Process Memory Usage : {:.2f} % - [".format(process_info.uss / init_free_memory*100) 
          + "{:.3f} GB / {:.3f} GB".format(process_info.uss * scale, init_free_memory * scale), end="]\n")
    
    print("Buffer  Memory Usage : {:.2f} % - [".format((process_info.uss - init_process_memory) / (init_free_memory - init_process_memory)*100) 
          + "{:.3f} GB / {:.3f} GB".format((process_info.uss - init_process_memory) * scale, (init_free_memory - init_process_memory) * scale), end="]\n")
    
    system_memory_percent = system_memory_info.used / system_memory_info.total * 100
    print("Total Memory Usage   : {:.2f} % - [".format(system_memory_percent) 
          + "{:.3f} GB / {:.3f} GB".format(system_memory_info.used * scale, system_memory_info.total * scale), end="]\n")
    
    cpu_usage_percent = psutil.cpu_percent()
    gpu_usage_percent = gpu.memoryUsed / gpu.memoryTotal * 100
    # print("CPU Usage            : {:.2f} % ".format(cpu_usage_percent))
    # print("GPU Usage            : {:.2f} % - [".format(gpu_usage_percent) + "{:.1f} MB / {:.1f} MB".format(gpu.memoryUsed, gpu.memoryTotal), end="]\n")

    if if_bar:
        print("Total Memory Usage   : {:.2f} % - [".format(system_memory_percent),
              "▓" * (int(system_memory_percent) // 2), "-"*((100 - int(system_memory_percent)) // 2), end="]\n")
        print("CPU Usage            : {:.2f} % - [".format(cpu_usage_percent),                   
              "▓" * (int(cpu_usage_percent) // 2), "-"*((100 - int(cpu_usage_percent)) // 2), end="]\n")
        print("GPU Usage            : {:.2f} % - [".format(gpu_usage_percent),                 
              "▓" * (int(gpu_usage_percent) // 2), "-"*((100 - int(gpu_usage_percent)) // 2), end="]\n")
    

class CoarseSimulator():
    def __init__(self, pixel_per_meter, max_acc=0, max_ang_acc=0):
        super(CoarseSimulator, self).__init__()
        self.max_acc = max_acc  # m/s^2
        self.max_ang_acc = max_ang_acc  # rad/s^2
        self.pixel_per_meter = pixel_per_meter
    
    def predict_state(self, vel, ang_vel, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        for _ in range(pre_step):
            x = - vel * np.cos(th) * dt  * self.pixel_per_meter + x
            y = - vel * np.sin(th) * dt  * self.pixel_per_meter + y
            th = ang_vel * dt + th

            next_xs.append(x)
            next_ys.append(y)
            next_ths.append(th)
            
        return next_xs, next_ys, next_ths
    
    
class TrajImage():
    def __init__(self, img_range, scale, idx, box_list, circle_list, name=""):
        
        self.scale = scale
        self.img_range = img_range
        self.name = name
        self.idx = idx
        
        self.box_list = box_list
        self.circle_list = circle_list
        
        self.colors = COLORS
        
        self.traj_img = np.ones((img_range, img_range, 3))
    
    
    def pos_scale(self, pos_list):
        return (- np.array(pos_list) * self.scale + self.img_range / 2).astype(np.int32)
        
    
    def draw_box(self, pos_list, color_index=-1, thickness=-1):
        pos_list = self.pos_scale(pos_list)
        num = int(len(pos_list) / 2)
        for i in range(num):
            cv2.rectangle(self.traj_img, 
                          (pos_list[2*i+1] - int(self.scale / 2), 
                           pos_list[2*i] - int(self.scale / 2)), 
                          (pos_list[2*i+1] + int(self.scale / 2), 
                           pos_list[2*i] + int(self.scale / 2)), 
                          self.colors[color_index],
                          thickness)
            
            
    def draw_circle(self, pos_list, color_index=-1, radius=0.5, thickness=-1, if_multi=False):
        pos_list = self.pos_scale(pos_list)
        num = int(len(pos_list) / 2)
        for i in range(num):
            idx = i if if_multi else color_index
            cv2.circle(self.traj_img, 
                       (pos_list[2*i+1], pos_list[2*i]), 
                       int(self.scale * radius), self.colors[idx], thickness)

    
    def init_img(self):
        self.traj_img = np.ones((self.img_range, self.img_range, 3))
        self.draw_box(self.box_list)
        self.draw_circle(self.circle_list)
            
    
    def viz(self):
        cv2.imshow('traj_img' + self.name, self.traj_img)
        cv2.waitKey(10)
        
        
    def get(self):
        return self.traj_img[:]
        
        
    def save(self):
        cv2.imwrite('./traj_img_' + self.name + '.png', self.traj_img * 255)
        

####################################
#              System
####################################     
        
def process_info():
    print('=============================>>>  Main  <<<=============================')
    print('parent process id : ', os.getppid())
    print('process id        : ', os.getpid())
    # print('ros port          : ', os.environ['ROS_MASTER_URI'])

    system_memory_info = psutil.virtual_memory()
    init_free_memory = system_memory_info.total - system_memory_info.used
    init_process_memory = psutil.Process(os.getpid()).memory_full_info().uss
    
    return init_free_memory, init_process_memory
    

####################################
#      Alternative Functions
####################################

def Lidar_Transformation_List(lidar_input, 
                              current_x, current_y, current_yaw, 
                              target_x, target_y, target_yaw, 
                              laser_num, laser_range,
                              scale=1):
    lidar_current = lidar_input
    current_x = current_x
    current_y = current_y
    current_yaw = current_yaw
    target_x = target_x
    target_y = target_y
    target_yaw = target_yaw

    index_xy = np.linspace(0, laser_num, laser_num)
    x_current = lidar_current * np.sin(index_xy / laser_num * math.pi).astype(np.float64)
    y_current = lidar_current * np.cos((1 - index_xy / laser_num) * math.pi).astype(np.float64)
    z_current = np.zeros_like(x_current)
    ones = np.ones_like(x_current)
    coordinates_current = np.stack([x_current, y_current, z_current, ones], axis=0)

    current_reference_x = current_x
    current_reference_y = current_y
    current_reference_yaw = current_yaw
    target_reference_x = target_x
    target_reference_y = target_y
    target_reference_yaw = target_yaw

    T_target_relative_to_world = np.array(
        [[math.cos(target_reference_yaw), -math.sin(target_reference_yaw), 0, target_reference_x],
         [math.sin(target_reference_yaw), math.cos(target_reference_yaw), 0, target_reference_y],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

    T_current_relative_to_world = np.array(
        [[math.cos(current_reference_yaw), -math.sin(current_reference_yaw), 0, current_reference_x],
         [math.sin(current_reference_yaw), math.cos(current_reference_yaw), 0, current_reference_y],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)
    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    lidar_length = np.sqrt(x_target * x_target + y_target * y_target) # 0 ~ LASER_RANGE
    lidar_angle = np.arctan2(y_target, x_target) / math.pi * 180 # -pi ~ pi => -180 ~ 180
    
    flag_in_fov = (lidar_angle > -90) & (lidar_angle < 90) & (lidar_current < (laser_range - 0.1)) # -180 ~ 180 => -90 ~ 90

    lidar_length = lidar_length[flag_in_fov]
    lidar_angle = np.floor(lidar_angle[flag_in_fov] + laser_num / 2 - 1).astype(np.int32) # -90 ~ 90 => 0 ~ laser_num

    lidar_output = np.ones(laser_num)
    lidar_output[lidar_angle] = lidar_length / laser_range

    lidar_output = lidar_output[::-1] # reverse

    return lidar_output


def Lidar_Transformation_Square(lidar_input, 
                                current_x, current_y, current_yaw, 
                                target_x, target_y, target_yaw,
                                laser_num, laser_range, img_range):

    lidar_current = np.asarray(lidar_input)

    index_xy = np.linspace(0,laser_num,laser_num)
    # x_current = lidar_current * np.sin(index_xy / laser_num * math.pi).astype(np.float64)
    # y_current = lidar_current * np.cos((1- index_xy / laser_num) * math.pi).astype(np.float64)
    x_current = lidar_current * np.cos(index_xy / laser_num * 2 * math.pi - math.pi).astype(np.float64)
    y_current = lidar_current * np.sin(index_xy / laser_num * 2 * math.pi - math.pi).astype(np.float64)


    z_current = np.zeros_like(x_current)
    ones = np.ones_like(x_current)
    coordinates_current = np.stack([x_current, y_current, z_current, ones], axis=0)

    current_reference_x = current_x
    current_reference_y = current_y
    current_reference_yaw = current_yaw
    target_reference_x = target_x
    target_reference_y = target_y
    target_reference_yaw = target_yaw

    # 坐标系转换：将当前坐标系下的坐标位置经由世界坐标系转到目标坐标系下
    T_target_relative_to_world = np.array([[math.cos(target_reference_yaw), -math.sin(target_reference_yaw), 0, target_reference_x],
                                            [math.sin(target_reference_yaw),  math.cos(target_reference_yaw), 0, target_reference_y],
                                            [                             0,                               0, 1,                  0],
                                            [                             0,                               0, 0,                  1]])

    T_current_relative_to_world = np.array([[math.cos(current_reference_yaw), -math.sin(current_reference_yaw), 0, current_reference_x],
                                            [math.sin(current_reference_yaw),  math.cos(current_reference_yaw), 0, current_reference_y],
                                            [                              0,                                 0, 1,                   0],
                                            [                              0,                                 0, 0,                   1]])

    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)

    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    # 转到图像坐标系下
    image_x = np.floor(img_range / 2.0 - 1.0 - x_target / laser_range * img_range / 2.0).astype(np.int32)
    image_y = np.floor(img_range / 2.0 - 1.0 - y_target / laser_range * img_range / 2.0).astype(np.int32)

    image_x[(image_x < 0) | (image_x > (img_range - 1)) | (lidar_current > (laser_range - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (img_range - 1)) | (lidar_current > (laser_range - 0.2))] = 0


    return image_x, image_y



def map2points(args, obs_map):
    
    position = np.where(obs_map > 0.8)
    arg_h = position[0]
    arg_w = position[1]
    p_x = args.img_range / 2 - arg_h
    p_y = args.img_range / 2 - arg_w
    
    p_x = p_x[np.where(p_x>0)]
    p_y = p_y[np.where(p_x>0)]

    # Nearest Points
    laser_len = 180
    laser = np.ones([3, laser_len]) * args.img_range / 2
    laser[2, :] = 1e6
    
    for i in range(len(p_x)):
        tmp_angle = math.atan2(p_y[i], p_x[i]) # -pi /2 ~ pi / 2
        int_angle = math.floor((tmp_angle + math.pi / 2) / math.pi * laser_len)

        dis2 = p_x[i] ** 2 + p_y[i] ** 2
        if(dis2 < laser[2, int_angle]):
            laser[0, int_angle] = p_x[i]
            laser[1, int_angle] = p_y[i]
            laser[2, int_angle] = dis2
            
    points = (laser[0:2].T).astype(np.float32)
    
    # DBSCAN
    # points = np.vstack([p_x,p_y]).astype(np.float32)
    # points = points.T
    # C = DBSCAN.DBSCAN_points(points)
    
    return points

def init_RL(args, rl_args, experiment):
    from policy.rl_algorithms.sac_graph import SAC as RL_Policy
    policy = RL_Policy(rl_args)
    if not args.graph_train:  # test阶段
        pre_policy = '{0}/{1}/policy/{2}/{3}'.format(args.root, args.model_file_name, experiment, args.graph_pre_model)
        policy.load(pre_policy)
    elif args.graph_pre_model > 0: # 训练阶段并且之前训练过一段时间
        pre_policy = '{0}/{1}/policy/{2}/{3}'.format(args.root, args.model_file_name, experiment, args.graph_pre_model)
        policy.load(pre_policy)
    return policy