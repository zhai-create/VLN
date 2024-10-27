# Navigation Module
## Navigate the robot to the sub-goal.

* topo_planner.py
    * class TopoPlanner(object):
        * self.prior_node_ls: Nodes involved in stitching.
        * self.topo_graph: Current topo-graph.
        * self.action_node: Selected action node based on RL policy.
        * self.rela_object_cx: Action node rela_cx.
        * self.rela_object_cy: Action node rela_cy.
        * self.object_parent_node: The parent node of the action node.
        * self.sub_map_node: The parent node to which the sub map used for local navigation belongs.
        * self.state_flag: Topo_planner state flag.
        * self.remain_nodes: Remaining no-navigated nodes on the topo-path.
        * self.origin_len_remain: Remain_nodes length before updating topo-path.
        * def get_topo_path(self):
            * Get the topo node_path, node list for map stitching, remain node list
        * def sub_map_stitching(self):
            * Sub map stitching
            * return obstacle_map: stitching map based on the nodes in the prior_node_ls
        * def get_start_end_point(self):
            * Get the start point, end point, stitching map for local planning
            * return start point: start pos for local planning
            * return end point: end pos for local planning
            * return stitching map: Map used for local planning

* local_planner.py
    * class LocalPlanner(object):
        * self.stitching_map: Map used for local planning.
        * self.start_point: start_pos for local planning.
        * self.end_point: end_pos for local planning.
        * self.state_flag: Topo_planner state flag.
        * self.action_node: selected sub-goal
        * self.topo_graph
        * self.sub_map_node: The node to which the submap belongs.
        * def get_local_path(self):
            * Get the local path based on rrt or astar
            * return rrt_path: local path
        * def update_local_path(self, topo_planner, next_action, rrt_path, rotate_cnt, temp_action_buffer):
            * Update local path and next action.
            * param topo_planner, next_action, rrt_path, rotate_cnt, temp_action_buffer
            * return next_action, rrt_path, rotate_cnt, temp_action_buffer

* RRTSTAR.py
    * class RRTNode:
        * Node for RRT
    * class RRTStar:
        * RRT STAR Planner

* ASTAR.py
    * class Map(object):
        * self.data: Map with starting and ending points in free areas.
        * startx, starty: Start point.
        * endx, endy: End poiny.
        * def get_free_obstacle_map(self, obstacles, position, delta):
            * Change the start and end points of the map to free areas.
            * param obstacles: origin map
            * param position: start point or end point.
            * param delta: free area delta.
            * return res_obstacles: Result of the free map.

    * class Node(object):
        * x, y: Node pos.
        * g: Already cost.
        * h: Potention cost.
        * father: parent node.
        * def find_nearest_grid(self, mapdata, temp_x, temp_y):
            * Get the nearest no-free area.
            * param mapdata: mao for astar.
            * param temp_x, temp_y: Coordinates to be investigated.
            * return min_dis: Distance to the nearest no-free area.
        * def get_new_cost(self, min_dis):
            * Get the additional cost.
            * param min_dis: min dis to the no-free area.
            * return additional cost
        * def getNeighbor(self,mapdata,endx,endy):
            * Get the neighbor nodes.
            * param mapdata: map for astar.
            * return endx, endy: end point.
        * def hasNode(self,worklist):
        * def changeG(self,worklist):

    * def getKeyforSort(element:Node):
        * Get the complet cost funcion value.
        * param Node: current astar node.
        * return result: complet cost funcion value.

    * def astar(workMap):
        * Astar planning.
        * param workMap: map for astar.
        * return result: atsar path.

* action_pub.py
    * def normalize_angle_list(angle_list):
        * Transform the angle_list to [-180, 180]
        * param angle_list: List of arbitrary angles to be converted
        * return normalized_angles: angle list between [-180, 180]
    * def normalize_angle_to_180(angle):
        * Transform the angle to [-180, 180]
        * param angle: Arbitrary angles to be converted
        * return normalized_angle: angle between [-180, 180]
    * def find_possible_direction(cur_dir, goal_dir):
        * Get the real goal turn-angle
        * param cur_dir: current turn-angle
        * param goal_dir: goal turn-angle
        * return candidate: Real goal turn-angle
    * def calculate_angle(A, B, C):
        * Get the angle that the robot rotates around the next path point
        * param A: last path node
        * param B: next path node
        * param C: current robot's pos
        * return angle_deg: Angle that robot rotate around the next node path
    * def choose_action(action_category, state_flag, last_action, path, cur_rela_loc, cur_rela_dir):
        * Get the next action and update path based on path and current_pos (new version)
        * param action_category: action node_type
        * param state_flag: topo_planner state flag
        * param last_action: last selected action
        * param path: last updated rrt node_path
        * param cur_rela_loc: current robot relative loc
        * param cur_rela_dir: current robot relative dir
        * return path: updated node_path
        * return selected action
    * def choose_action_origin(path, cur_rela_loc, cur_rela_dir):
        * Get the next action and update path based on path and current_pos (old version)
        * param path: last updated rrt node_path
        * param cur_rela_loc: current robot relative loc
        * param cur_rela_dir: current robot relative dir
        * return path: updated node_path
        * return selected action

* habitat_action.py
    * class HabitatAction:
        * count_steps: all step num.
        * front_steps: front step num.
        * walk_path_meter: The path meter of robot walking.
        * this_episode_short_dis: init min_distance to goal.
        * def reset(habitat_env):
            * Reset the static attributes.
            * param habitat_env
        * def set_habitat_action(action_name, topo_graph):
            * Process the habitat action and get the robot's pos after habitat_action.
            * param action_name: 'f' or 'l' or 'r'
            * param topo_graph
            * return habitat_action: action in the habitat_env.

* sub_goal_reach.py
    * class SubgoalReach:
        * next_action: Selected next action.
        * rotate_cnt: The number of times the robot shakes left and right.
        * temp_action_buffer: Next action buffer.
        * last_sim_location: last loc in the habitat.
        * path_block_ls: List of robot walking path mater.
        * def reset():
            * Reset the static attributes.
        * def is_block(habitat_env):
            * Determine if the robot is block.
            * param habitat_env
            * return flag: True or False
        * def achieved__action_node(topo_graph, action_node):
            * When the robot achieves the sub-goal, delete the action node in the topo-graph.
            * param topo_graph
            * param action_node
        * def go_to_sub_goal(topo_graph, action_node, habitat_env, object_goal):
            * Go to the selected action node pos.
            * param topo_graph
            * param action_node
            * param habitat_env
            * param object_goal

* tools.py
    * def get_nearest_grid(end_point, temp_ghost_obstacle_map, action_category):
        * Get the nearest free area.
        * param end_point: frontier pos or intention pos
        * param temp_ghost_obstacle_map: map.
        * param action_category: action_node type.
        * return min_grid_x, min_grid_y: Nearest free map area.

    * def get_l2_distance(x1, x2, y1, y2):
        * Get the euclidean distance.

    * def get_sim_location(habitat_env):
        * Returns x, y, o pose of the agent in the Habitat simulator.

    * def get_rel_pose_change(pos2, pos1):
        * Get the pos change from pos1 and pos2.

    * def get_pose_change(habitat_env, last_sim_location):
        * Returns dx, dy, do pose change of the agent relative to the last timestep.

    * def is_in_free_grid(temp_node, current_node, rela_cx, rela_cy):
        * Determine whether the current robot is located in the free area of temp_node.
        * param temp_node: Node to be inspected
        * param current_node
        * param rela_cx, rela_cy
        * return flag: True or False.

    * def plot_map(obstacles, path, suc):
        * Utility: Show the rrt path and its map.

    * def is_temp_node_see(temp_node, current_node, rela_cx, rela_cy, rela_turn):
        * Determine whether the center of the current node is within the range of the local grid of the temp_node.
    


