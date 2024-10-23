import os
import cv2
import argparse
import csv
from tqdm import tqdm
import numpy as np

import habitat
from env_tools.data_utils import hm3d_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FINISH="f"
FORWARD="w"
LEFT="a"
RIGHT="d"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=100)
    parser.add_argument("--mapper_resolution",type=float,default=0.05)
    parser.add_argument("--path_resolution",type=float,default=0.2)
    parser.add_argument("--path_scale",type=int,default=5)
    return parser.parse_known_args()[0]

from perception.frontier_utils import predict_frontier
from perception.laser_utils import get_laser_point
from perception.tools import fix_depth, get_rgb_image
from graph.node_utils import Node

if __name__ == "__main__":
    args = get_args()
    habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
    habitat_env = habitat.Env(config=habitat_config)


    for i in tqdm(range(args.eval_episodes)):
        observations = habitat_env.reset()
        # rgb_show
        # obs_rgb = observations["rgb"]
        # cv2.imshow("obs_rgb", obs_rgb)

        rgb_1 = get_rgb_image(habitat_env, 1)
        rgb_2 = get_rgb_image(habitat_env, 2)
        rgb_3 = get_rgb_image(habitat_env, 3)
        rgb_4 = get_rgb_image(habitat_env, 4)


        cv2.imshow("rgb_1", rgb_1)
        cv2.imshow("rgb_2", rgb_2)
        cv2.imshow("rgb_3", rgb_3)
        cv2.imshow("rgb_4", rgb_4)

        # depth_show
        obs_depth = observations["depth"]
        d_for_show = cv2.resize(obs_depth, None, fx=1.5, fy=1.5)
        cv2.imshow("d_for_show", d_for_show)

        obs_depth = fix_depth(obs_depth)
        point_for_close_loop_detection, laser_2d_filtered, laser_2d_filtered_angle = \
        get_laser_point(obs_depth)

        print("point_for_close_loop_detection:{}\n".format(point_for_close_loop_detection))
        print("laser_2d_filtered:{}\n".format(laser_2d_filtered))
        print("laser_2d_filtered_angle:{}\n".format(laser_2d_filtered_angle))
        
        node1 = Node(node_type="explored_node", name="0", pc=point_for_close_loop_detection)
        node1.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, np.array([0,0]), 0.0)

        occu_for_show = cv2.resize(node1.occupancy_map.astype(np.float64), None, fx=1, fy=1)
        cv2.imshow("occupancy_map", occu_for_show)

        candidate_frontier_arr = predict_frontier(0.1, laser_2d_filtered, laser_2d_filtered_angle)

        print("candidate_frontier_arr:", candidate_frontier_arr)

        # metric
        habitat_metric = habitat_env.get_metrics()

        while True:
            keystroke = cv2.waitKey(0)
            if(keystroke == ord(FORWARD)):
                habitat_action = HabitatSimActions.move_forward
            elif(keystroke == ord(LEFT)):
                habitat_action = HabitatSimActions.turn_left
            elif(keystroke == ord(RIGHT)):
                habitat_action = HabitatSimActions.turn_right
            elif(keystroke == ord(FINISH)):
                habitat_action = HabitatSimActions.stop
                habitat_env.step(habitat_action)
                break
            else:
                print("invalid_key")
                continue
            observations = habitat_env.step(habitat_action)

            # rgb_show
            # obs_rgb = observations["rgb"]
            # cv2.imshow("obs_rgb", obs_rgb)


            rgb_1 = get_rgb_image(habitat_env, 1)
            rgb_2 = get_rgb_image(habitat_env, 2)
            rgb_3 = get_rgb_image(habitat_env, 3)
            rgb_4 = get_rgb_image(habitat_env, 4)


            cv2.imshow("rgb_1", rgb_1)
            cv2.imshow("rgb_2", rgb_2)
            cv2.imshow("rgb_3", rgb_3)
            cv2.imshow("rgb_4", rgb_4)

            # depth_show
            obs_depth = observations["depth"]
            d_for_show = cv2.resize(obs_depth, None, fx=1.5, fy=1.5)
            cv2.imshow("d_for_show", d_for_show)

            obs_depth = fix_depth(obs_depth)

            point_for_close_loop_detection, laser_2d_filtered, laser_2d_filtered_angle = \
            get_laser_point(obs_depth)

            print("point_for_close_loop_detection:{}\n".format(point_for_close_loop_detection))
            print("laser_2d_filtered:{}\n".format(laser_2d_filtered))
            print("laser_2d_filtered_angle:{}\n".format(laser_2d_filtered_angle))

            node1.update_occupancy(laser_2d_filtered, laser_2d_filtered_angle, np.array([0,0]), 0.0)

            occu_for_show = cv2.resize(node1.occupancy_map.astype(np.float64), None, fx=1, fy=1)
            cv2.imshow("occupancy_map", occu_for_show)

            candidate_frontier_arr = predict_frontier(0.1, laser_2d_filtered, laser_2d_filtered_angle)

            print("candidate_frontier_arr:", candidate_frontier_arr)
            
            # metric
            habitat_metric = habitat_env.get_metrics()

            # agent_pos
            c_x, c_y = observations["agent_position"][2], observations["agent_position"][0]
            c_z = observations["agent_position"][1]
            
            print("cx{}, cy:{}, cz:{}\n".format(c_x, c_y, c_z))
            print("dis_to_goal:{}\n".format(habitat_metric['distance_to_goal']))
            print("----habitat_metric----", habitat_metric.keys())
            print("----observation_keys----", observations.keys())
