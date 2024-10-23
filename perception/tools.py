import numpy as np
import quaternion
from scipy.ndimage import zoom

from perception.arguments import args, pre_depth, ta_ls_array
from graph.tools import get_absolute_pos

def fix_depth(depth):
    """
    Fix the sensor depth
    :param depth: original sensor depth
    :return depth: depth with real dis
    """
    depth = depth / pre_depth.data[:, :, 4:]
    return depth

def resize_matrix(matrix, new_shape):
    """
    Reduce the size of the corresponding matrix and output the values of each position in the matrix according to the relative ratio of the positions of the values in the original matrix on the rows and columns
    :param matrix: the imput 2d matrix(element: 0 or 1)
    :param new_shape: the output matrix shape (new_row_num, new_col_num)
    :return: resized matrix
    """
    # 计算缩放比例
    zoom_factors = (new_shape[0] / matrix.shape[0], new_shape[1] / matrix.shape[1])
    # 使用缩放比例缩小矩阵尺寸
    resized_matrix = zoom(matrix, zoom_factors, order=1)
    
    return resized_matrix


def depth_estimation(large_mask, depth):
    """
    Get the relative pos of the detected object
    :param large_mask: mask result corresponding to the panoramic depth
    :param depth: fixed depth
    :return res_depth_2d_cx, res_depth_2d_cy: relative pos of the object corresponding to the current robot
    """
    min_threshold = 0.1
    depth_2d = depth * np.cos(pre_depth.data[:, :, 2:3]) * args.depth_scale # 单位：meter

    lower_depth_height = int(args.depth_height/4)
    upper_depth_height = int(3*args.depth_height/4)
    partial_depth_2d = depth_2d[lower_depth_height:upper_depth_height, :, 0]

    num_large_mask = np.where(large_mask, 1, 0)
    num_large_mask_after_resize = resize_matrix(num_large_mask, new_shape=(partial_depth_2d.shape[0], partial_depth_2d.shape[1]))
    
    target_depth = np.multiply(num_large_mask_after_resize, partial_depth_2d)
    sum_non_zero = np.sum(target_depth)
    count_non_zero = np.count_nonzero(target_depth)
    average_depth = sum_non_zero / count_non_zero

    non_zero_indices = np.nonzero(target_depth)
    x_indices, y_indices = non_zero_indices[0], non_zero_indices[1]
    average_y = np.mean(y_indices)

    ta_angle = ta_ls_array[int(average_y)]
    res_depth_2d_cx = average_depth * np.cos(ta_angle)
    res_depth_2d_cy = average_depth * np.sin(ta_angle)

    return res_depth_2d_cx, res_depth_2d_cy

def sam_show_mask(mask, image0):
    """
        Utility: Show the mask on the rgb image
    """
    value = 0  # 0 for background
    for temp_i in range(mask.shape[0]):
        for temp_j in range(mask.shape[1]):
            if(mask[temp_i][temp_j]==True):
                image0[temp_i][temp_j][0] = int(image0[temp_i][temp_j][0] + 0.3 * 0)
                image0[temp_i][temp_j][1] = int(image0[temp_i][temp_j][1] + 0.3 * 0)
                image0[temp_i][temp_j][2] = int(255)
    return image0

def transform_rgb_bgr(image):
    """
    Transform rgb to bgr
    :param image: rgb
    :return image: bgr
    """
    return image[:, :, [2, 1, 0]]

def get_rgb_image(env, turn_id):
    #     1
    # 2        4
    #     3
    rgb = None
    if(turn_id==1):
        dis, angle = 0, 0
        p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
        state = env._sim.get_agent_state(0)
        translation = state.position # 1: right; 2: up; 3: back
        rotation = state.rotation # anti-clockwise / up-righthand principle
        euler = quaternion.as_euler_angles(rotation)
        if euler[0]!=0:
            euler[1] = 2*euler[0]-euler[1]
            euler[0] = 0
            euler[2] = 0
        ref_true_loc = np.array([-translation[2], translation[0]])
        ref_true_dir = euler[1]
        rotation = quaternion.from_euler_angles(euler)

        p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
        goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
        obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
        rgb = transform_rgb_bgr(obs["rgb"])
        # print(state.position, type(state.position))
        # print(rotation, type(rotation))
        # print("euler1:", euler)
    elif(turn_id==2):
        dis, angle = 0, 0.5*np.pi
        p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
        state = env._sim.get_agent_state(0)
        translation = state.position # 1: right; 2: up; 3: back
        rotation = state.rotation # anti-clockwise / up-righthand principle
        euler = quaternion.as_euler_angles(rotation)
        if euler[0]!=0:
            euler[1] = 2*euler[0]-euler[1]
            euler[0] = 0
            euler[2] = 0
        euler[1]+=0.5*np.pi
        ref_true_loc = np.array([-translation[2], translation[0]])
        ref_true_dir = euler[1]
        rotation = quaternion.from_euler_angles(euler)
        p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
        goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
        obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
        rgb = transform_rgb_bgr(obs["rgb"])
        # print(state.position, type(state.position))
        # print(rotation, type(rotation))
        # print("euler2:", euler)
    elif(turn_id==3):
        dis, angle = 0, np.pi
        p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
        state = env._sim.get_agent_state(0)
        translation = state.position # 1: right; 2: up; 3: back
        rotation = state.rotation # anti-clockwise / up-righthand principle
        euler = quaternion.as_euler_angles(rotation)
        if euler[0]!=0:
            euler[1] = 2*euler[0]-euler[1]
            euler[0] = 0
            euler[2] = 0
        euler[1]+=np.pi
        ref_true_loc = np.array([-translation[2], translation[0]])
        ref_true_dir = euler[1]
        rotation = quaternion.from_euler_angles(euler)
        p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
        goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
        obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
        rgb = transform_rgb_bgr(obs["rgb"])
        # print(state.position, type(state.position))
        # print(rotation, type(rotation))
        # print("euler3:", euler)
    elif(turn_id==4):
        dis, angle = 0, -0.5*np.pi
        p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
        state = env._sim.get_agent_state(0)
        translation = state.position # 1: right; 2: up; 3: back
        rotation = state.rotation # anti-clockwise / up-righthand principle
        euler = quaternion.as_euler_angles(rotation)
        if euler[0]!=0:
            euler[1] = 2*euler[0]-euler[1]
            euler[0] = 0
            euler[2] = 0
        euler[1]-=0.5*np.pi
        ref_true_loc = np.array([-translation[2], translation[0]])
        ref_true_dir = euler[1]
        rotation = quaternion.from_euler_angles(euler)
        p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
        goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
        obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
        rgb = transform_rgb_bgr(obs["rgb"])
        # print(state.position, type(state.position))
        # print(rotation, type(rotation))
        # print("euler4:", euler)
    return rgb


def get_rgb_image_ls(env):
    """
    Get the image list
    :param env: habitat_env
    :return image_ls: [rgb_1, rgb_2, rgb_3, rgb_4]
    """
    #     1
    # 2        4
    #     3
    rgb_ls = []
    for turn_id in range(1, 5):
        if(turn_id==1):
            dis, angle = 0, 0
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)

            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = transform_rgb_bgr(obs["rgb"])
        elif(turn_id==2):
            dis, angle = 0, 0.5*np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]+=0.5*np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = transform_rgb_bgr(obs["rgb"])
        elif(turn_id==3):
            dis, angle = 0, np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]+=np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = transform_rgb_bgr(obs["rgb"])
        elif(turn_id==4):
            dis, angle = 0, -0.5*np.pi
            p_ref_loc = np.array([dis*np.sin(angle), dis*np.cos(angle)])
            state = env._sim.get_agent_state(0)
            translation = state.position # 1: right; 2: up; 3: back
            rotation = state.rotation # anti-clockwise / up-righthand principle
            euler = quaternion.as_euler_angles(rotation)
            if euler[0]!=0:
                euler[1] = 2*euler[0]-euler[1]
                euler[0] = 0
                euler[2] = 0
            euler[1]-=0.5*np.pi
            ref_true_loc = np.array([-translation[2], translation[0]])
            ref_true_dir = euler[1]
            rotation = quaternion.from_euler_angles(euler)
            p_true_loc = get_absolute_pos(p_ref_loc, ref_true_loc, ref_true_dir)
            goal_position = np.array([p_true_loc[1], translation[1], -p_true_loc[0]])
            obs = env._sim.get_observations_at(position=goal_position, rotation=rotation, keep_agent_at_new_pose=False)
            rgb = transform_rgb_bgr(obs["rgb"])
        rgb_ls.append(rgb)
    return rgb_ls