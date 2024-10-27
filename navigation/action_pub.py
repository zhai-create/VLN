import numpy as np

from navigation.tools import get_absolute_pos_world
from navigation.arguments import act_num_str_map
from navigation.arguments import args

from graph.arguments import args as graph_args
from perception.arguments import args as perception_args

half_len = (int)(perception_args.depth_scale/graph_args.resolution)

def choose_action(local_path, sub_map_node, habitat_env, habitat_planner):
    if len(local_path) < args.path_length_thre: # 路径较短
        waypoint_grid = local_path[-1]
    else: # 路径较长
        waypoint_grid = local_path[args.path_length_thre-1] # 我们是9

    waypoint_x = -(waypoint_grid[0] - half_len) * graph_args.resolution
    waypoint_y = (waypoint_grid[1] - half_len) * graph_args.resolution
    waypoint = np.array([waypoint_x, waypoint_y])
    
    current_position = np.array([habitat_env._sim.get_agent_state(0).position[2], habitat_env._sim.get_agent_state(0).position[0]])
    world_waypoint = get_absolute_pos_world(waypoint[0], waypoint[1], sub_map_node.world_cx, sub_map_node.world_cy, sub_map_node.world_turn)
    world_waypoint = np.array([world_waypoint[0], world_waypoint[1]])
    
    print("-----> waypoint_grid <-----", waypoint_grid)
    
    to_target_distance = np.sqrt(np.sum(np.square(current_position - world_waypoint))) # 计算当前机器人的位置与下一个路径点指点的距离(包括高度)
    print("=====> to_target_distance <=====", to_target_distance)
    
    
    if to_target_distance < args.to_target_distance_thre and len(local_path) > 0: # 当agent距离waypoint较近时，更新waypoint 和 local_path
        local_path = local_path[min(args.path_length_thre, len(local_path)-1):] # 更新local_path
        if len(local_path) < args.short_path_length_thre: 
            waypoint_grid = local_path[-1]
        else:
            waypoint_grid = local_path[args.short_path_length_thre-1]

        waypoint_x = -(waypoint_grid[0] - half_len) * graph_args.resolution
        waypoint_y = (waypoint_grid[1] - half_len) * graph_args.resolution
        waypoint = np.array([waypoint_x, waypoint_y])

    print("=====> waypoint_grid <=====", waypoint_grid)

    pid_waypoint = get_absolute_pos_world(waypoint[0], waypoint[1], sub_map_node.world_cx, sub_map_node.world_cy, sub_map_node.world_turn)
    pid_waypoint = np.array([pid_waypoint[1], habitat_env._sim.get_agent_state(0).position[1], pid_waypoint[0]]) # pid_waypoint: 下一个路径点在世界坐标系下的坐标
    
    habitat_act_num = habitat_planner.get_next_action(pid_waypoint)
    next_action = act_num_str_map[habitat_act_num]

    return local_path, next_action
    
        
