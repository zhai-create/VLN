import os
import cv2
import argparse
import csv
from tqdm import tqdm
import numpy as np

import habitat
from env_tools.data_utils import hm3d_config
from env_tools.arguments import args as env_args
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

if __name__ == "__main__":
    args = get_args()
    habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
    habitat_env = habitat.Env(config=habitat_config)


    for i in tqdm(range(args.eval_episodes)):
        episode_steps = 0
        observations = habitat_env.reset()

        object_goal = env_args.object_ls[observations["objectgoal"][0]]
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

        

        candidate_frontier_arr = predict_frontier(0.1, laser_2d_filtered, laser_2d_filtered_angle)


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
            else:
                print("invalid_key")
                continue
            observations = habitat_env.step(habitat_action)
            episode_steps += 1

            print("episode_steps:", episode_steps)
            print("habitat_action:", habitat_action)
            print("habitat_env.episode_over:", habitat_env.episode_over)
            print("object_goal:", object_goal)

            # metric
            habitat_metric = habitat_env.get_metrics()
            print("habitat_metric:", habitat_metric)

            print("dis_to_goal:{}\n".format(habitat_metric['distance_to_goal']))

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




            candidate_frontier_arr = predict_frontier(0.1, laser_2d_filtered, laser_2d_filtered_angle)

            
            

            # agent_pos
            c_x, c_y = observations["agent_position"][2], observations["agent_position"][0]
            c_z = observations["agent_position"][1]
            
            # print("cx{}, cy:{}, cz:{}\n".format(c_x, c_y, c_z))
            # print("----habitat_metric----", habitat_metric.keys())
            # print("----observation_keys----", observations.keys())
