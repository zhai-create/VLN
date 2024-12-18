import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
import imageio


from habitat.utils.visualizations import maps
from typing import TYPE_CHECKING, Union, cast

from navigation.tools import get_absolute_pos_world
from graph.tools import get_current_world_pos

from perception.tools import get_rgb_image_ls, fix_depth

from vis_tools.arguments import args

def get_top_down_map(habitat_env):
    top_down_map = maps.get_topdown_map_from_sim(
        cast("HabitatSim", habitat_env.sim), map_resolution=1024
    )
    recolor_map = np.array(
        [[255, 255, 255], [200, 200, 200], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    cv2.imwrite("{}/top_down_map.png".format(args.pre_path), top_down_map)



def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    image = image[:,:,:3]
    return image

def plot_topomap_on_global_map(habitat_env, topo_graph, rl_graph, action_node):
    label_figure = plt.figure(2, figsize=(3, 5))
    plt.clf()
    ax = plt.gca()
    # 加载背景图片
    background_image = mpimg.imread("{}/top_down_map.png".format(args.pre_path)) 
    # 绘制背景图片
    ax.imshow(background_image)
    ax.set_xlim(0, background_image.shape[1])  # 设置x轴范围为0到图像宽度
    ax.set_ylim(background_image.shape[0], 0)

    G = nx.Graph()
    pos = {}
    color = {}

    world_cx, world_cy, world_cz, world_turn = get_current_world_pos(habitat_env)


    label_action_ls_index = 0
    label_temp_edge_index = 0
    label_new_edge_dict = {}
    graph_num_action_padding = 500
    selected_intention_node_ls = rl_graph.select_intention(topo_graph)
    
    for temp_node in topo_graph.all_nodes:
        if(temp_node.node_type=="explored_node"):
            tx, ty = maps.to_grid(
                temp_node.world_cx,
                temp_node.world_cy,
                (background_image.shape[0], background_image.shape[1]),
                sim=habitat_env.sim,
            )

            if temp_node.name == topo_graph.current_node.name:
                circle = plt.Circle((ty, tx), radius=30, color=(210/255,174/255,172/255), zorder=2)  # 设置圆圈的大小、颜色等
            else:
                circle = plt.Circle((ty, tx), radius=30, color=(142/255,165/255,200/255), zorder=2)  # 设置圆圈的大小、颜色等
            ax.add_patch(circle)
            G.add_node(temp_node.name, pos=(ty, tx))
            label_new_edge_dict.update({temp_node.name: label_temp_edge_index})
            label_temp_edge_index += 1

        elif(temp_node.node_type=="frontier_node"): # frontier
            if(label_action_ls_index>=graph_num_action_padding):
                continue
            res_loc_in_real_world = get_absolute_pos_world(temp_node.rela_cx, temp_node.rela_cy, temp_node.parent_node.world_cx, temp_node.parent_node.world_cy, temp_node.parent_node.world_turn)
            tx, ty = maps.to_grid(
                res_loc_in_real_world[0],
                res_loc_in_real_world[1],
                (background_image.shape[0], background_image.shape[1]),
                sim=habitat_env.sim,
            )
            
            if(action_node.name==temp_node.name):
                circle = plt.Circle((ty, tx), radius=25, color=(255/255,0/255,0/255), zorder=2)  # 设置圆圈的大小、颜色等
            else:
                circle = plt.Circle((ty, tx), radius=20, color=(161/255,125/255,180/255), zorder=2)  # 设置圆圈的大小、颜色等
            ax.add_patch(circle)
            G.add_node(temp_node.name, pos=(ty, tx))
            label_new_edge_dict.update({temp_node.name: label_temp_edge_index})
            label_temp_edge_index += 1
            label_action_ls_index += 1
            
        elif(temp_node.node_type=="intention_node" and ((temp_node in selected_intention_node_ls))): 
            if(label_action_ls_index>=graph_num_action_padding):
                continue
            res_loc_in_real_world = get_absolute_pos_world(temp_node.rela_cx, temp_node.rela_cy, temp_node.parent_node.world_cx, temp_node.parent_node.world_cy, temp_node.parent_node.world_turn)
            tx, ty = maps.to_grid(
                res_loc_in_real_world[0],
                res_loc_in_real_world[1],
                (background_image.shape[0], background_image.shape[1]),
                sim=habitat_env.sim,
            )

            if(action_node.name==temp_node.name):
                circle = plt.Circle((ty, tx), radius=25, color=(255/255,0/255,0/255), zorder=2)  # 设置圆圈的大小、颜色等
            else:
                circle = plt.Circle((ty, tx), radius=20, color=(0/255,255/255,0/255), zorder=2)  # 设置圆圈的大小、颜色等
            ax.add_patch(circle)
            G.add_node(temp_node.name, pos=(ty, tx))
            label_action_ls_index += 1
            label_new_edge_dict.update({temp_node.name: label_temp_edge_index})
            label_temp_edge_index += 1

    for temp_node in topo_graph.all_nodes:
        if(temp_node.name in label_new_edge_dict):
            if(temp_node.node_type=="explored_node"):
                for temp_neighbor_name in temp_node.neighbor:
                    G.add_edge(temp_neighbor_name, temp_node.name)
            else:
                G.add_edge(temp_node.parent_node.name, temp_node.name)
        else:
            continue

    current_tx, current_ty = maps.to_grid(
            world_cx,
            world_cy,
            (background_image.shape[0], background_image.shape[1]),
            sim=habitat_env.sim,
        )

    circle = plt.Circle((current_ty, current_tx), radius=30, color=(235/255,161/255,51/255), zorder=2)  # 设置圆圈的大小、颜色等
    ax.add_patch(circle)

    sub_goal_absolute_pos = get_absolute_pos_world(action_node.rela_cx, action_node.rela_cy, action_node.parent_node.world_cx, action_node.parent_node.world_cy, action_node.parent_node.world_turn)
    sub_goal_tx, sub_goal_ty = maps.to_grid(
        sub_goal_absolute_pos[0],
        sub_goal_absolute_pos[1],
        (background_image.shape[0], background_image.shape[1]),
        sim=habitat_env.sim,
    )

    circle = plt.Circle((sub_goal_ty, sub_goal_tx), radius=30, color=(255/255,0/255,0/255), zorder=2)  # 设置圆圈的大小、颜色等
    ax.add_patch(circle)

    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=(95/255, 95/255, 95/255))

    # 设置坐标轴不可见
    ax.set_axis_off()

    # 将plt转化为numpy数据
    canvas = fig2data(label_figure)
    return canvas


def init_mp4(pre_model, episode_index):
    robot_trajectory = []
    if(not os.path.exists('{}/'.format(args.pre_path))):
        os.makedirs('{}/'.format(args.pre_path))
    save_path = "{}/rgb_{:04d}_{:04d}.mp4".format(args.pre_path, pre_model, episode_index)
    video_writer = imageio.get_writer(
        save_path,
        codec="h264",
        fps=10,
        quality=None,
        pixelformat="yuv420p",
        bitrate=0,
        output_params=["-crf", "31"],
    )

    map_path = "{}/graph_{:04d}_{:04d}.mp4".format(args.pre_path, pre_model, episode_index)
    map_writer = imageio.get_writer(
        map_path,
        codec="h264",
        fps=10,
        quality=None,
        pixelformat="yuv420p",
        bitrate=0,
        output_params=["-crf", "31"],
    )
    return video_writer, map_writer


def save_mp4(video_writer, map_writer, habitat_env, topo_graph, rl_graph, action_node, object_goal):
    plt_topo_map = plot_topomap_on_global_map(habitat_env, topo_graph, rl_graph, action_node)

    rgb_image_ls = get_rgb_image_ls(habitat_env) # [1, 2, 3, 4]
    video_image = rgb_image_ls[0][..., ::-1]
    video_image = np.array(video_image, dtype=np.uint8)
    video_image = cv2.resize(video_image, None, fx=1.0, fy=1.0)
    cv2.rectangle(video_image, args.new_top_left, args.new_right_bottom, color=(0,0,0), thickness=-1)
    cv2.putText(video_image, 'Object Goal: '+object_goal, args.font_pos1, cv2.FONT_HERSHEY_SIMPLEX, args.font_size, (255, 255, 255), args.font_width)

    if(action_node.node_type=="frontier_node"):
        cv2.putText(video_image, "Go to frontier node", args.font_pos2, cv2.FONT_HERSHEY_SIMPLEX, args.font_size, (255, 255, 255), args.font_width)
    else:
        cv2.putText(video_image, "Go to intention node", args.font_pos2, cv2.FONT_HERSHEY_SIMPLEX, args.font_size, (255, 255, 255), args.font_width)

    video_writer.append_data(video_image)
    map_writer.append_data(plt_topo_map)


