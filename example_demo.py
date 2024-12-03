import os
import cv2
import argparse
import csv
from tqdm import tqdm
import numpy as np
import random

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
    parser.add_argument("--eval_episodes",type=int,default=1)
    parser.add_argument("--mapper_resolution",type=float,default=0.05)
    parser.add_argument("--path_resolution",type=float,default=0.2)
    parser.add_argument("--path_scale",type=int,default=5)
    return parser.parse_known_args()[0]

from perception.frontier_utils import predict_frontier
from perception.laser_utils import get_laser_point
from perception.tools import fix_depth, get_rgb_image

if __name__ == "__main__":
    args = get_args()
    args.eval_episodes = 80000
    habitat_config = hm3d_config(stage='train',episodes=1, max_steps=1000000)
    habitat_env = habitat.Env(config=habitat_config)

    print("len(env.episodes):", len(habitat_env.episodes))
    # # =====> select episodes <=====
    # random.seed(456)
    # id_dict = {
    #     # "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":["bed", "tv_monitor"]
    #     "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":["bed", "tv_monitor", "toilet", "sofa", "chair", "plant"][4]
    #     # "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":["plant"]
    # }
    # selected_episodes = []
    # for index, temp_episode in enumerate(habitat_env.episodes):
    #     if(temp_episode.scene_id in id_dict.keys()):
    #         if(temp_episode.object_category in id_dict[temp_episode.scene_id]):
    #             selected_episodes.append(temp_episode)
    # random.shuffle(selected_episodes)
    # # =====> select episodes <=====

    # =====> select episodes <=====
    random.seed(456)
    id_dict = {
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00669-DNWbUAJYsPy/DNWbUAJYsPy.basis.glb":["tv_monitor", "bed", "sofa", "chair", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00166-RaYrxWt5pR1/RaYrxWt5pR1.basis.glb":["tv_monitor", "toilet", "chair", "plant", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00404-QN2dRqwd84J/QN2dRqwd84J.basis.glb":["sofa", "bed", "plant", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00706-YHmAkqgwe2p/YHmAkqgwe2p.basis.glb":["bed", "toilet", "chair", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00324-DoSbsoo4EAg/DoSbsoo4EAg.basis.glb":["bed", "tv_monitor"],

        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00017-oEPjPNSPmzL/oEPjPNSPmzL.basis.glb":["bed", "tv_monitor", "toilet", "sofa", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00031-Wo6kuutE9i7/Wo6kuutE9i7.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00099-226REUyJh2K/226REUyJh2K.basis.glb":["bed", "tv_monitor"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00105-xWvSkKiWQpC/xWvSkKiWQpC.basis.glb":["tv_monitor", "toilet", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00250-U3oQjwTuMX8/U3oQjwTuMX8.basis.glb":["bed", "toilet", "sofa", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00251-wsAYBFtQaL7/wsAYBFtQaL7.basis.glb":["bed", "toilet", "sofa"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00254-YMNvYDhK8mB/YMNvYDhK8mB.basis.glb":["chair", "plant"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00255-NGyoyh91xXJ/NGyoyh91xXJ.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00323-yHLr6bvWsVm/yHLr6bvWsVm.basis.glb":["bed", "tv_monitor", "toilet"],
        "./dependencies/habitat-lab/data/scene_datasets/hm3d_v0.2/train/00327-xgLmjqzoAzF/xgLmjqzoAzF.basis.glb":["bed", "toilet", "chair"],

    }
    selected_episodes = []
    for index, temp_episode in enumerate(habitat_env.episodes):
        if(temp_episode.scene_id in id_dict.keys()):
            if(temp_episode.object_category in id_dict[temp_episode.scene_id]):
                selected_episodes.append(temp_episode)
    random.shuffle(selected_episodes)
    # =====> select episodes <=====


    for i in tqdm(range(args.eval_episodes)):
        episode_steps = 0
        habitat_env.episodes = [selected_episodes[i]]
        observations = habitat_env.reset()

        object_goal = env_args.object_ls[observations["objectgoal"][0]]
        # rgb_show
        # obs_rgb = observations["rgb"]
        # cv2.imshow("obs_rgb", obs_rgb)

        rgb_1 = get_rgb_image(habitat_env, 1)
        rgb_2 = get_rgb_image(habitat_env, 2)
        rgb_3 = get_rgb_image(habitat_env, 3)
        rgb_4 = get_rgb_image(habitat_env, 4)

        rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
        large_rgb = np.hstack((rgb_image_ls[2][:, int(rgb_image_ls[2].shape[1]//2):], rgb_image_ls[1], rgb_image_ls[0], rgb_image_ls[3], rgb_image_ls[2][:, :int(rgb_image_ls[2].shape[1]//2)]))
        large_rgb_for_show = cv2.resize(large_rgb, None, fx=0.5, fy=0.5)
        # cv2.imshow("large_rgb_for_show", large_rgb_for_show)
        cv2.imshow("rgb_1", rgb_1)
        
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

        while not habitat_env.episode_over:
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

            print("---------------------")
            print("episode_steps:", episode_steps)
            print("habitat_action:", habitat_action)
            print("habitat_env.episode_over:", habitat_env.episode_over)
            print("object_goal:", object_goal)

            # metric
            habitat_metric = habitat_env.get_metrics()
            print("habitat_metric:", habitat_metric)

            print("dis_to_goal:{}\n".format(habitat_metric['distance_to_goal']))
            print("====================")

            # rgb_show
            # obs_rgb = observations["rgb"]
            # cv2.imshow("obs_rgb", obs_rgb)


            rgb_1 = get_rgb_image(habitat_env, 1)
            rgb_2 = get_rgb_image(habitat_env, 2)
            rgb_3 = get_rgb_image(habitat_env, 3)
            rgb_4 = get_rgb_image(habitat_env, 4)

            rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
            large_rgb = np.hstack((rgb_image_ls[2][:, int(rgb_image_ls[2].shape[1]//2):], rgb_image_ls[1], rgb_image_ls[0], rgb_image_ls[3], rgb_image_ls[2][:, :int(rgb_image_ls[2].shape[1]//2)]))
            large_rgb_for_show = cv2.resize(large_rgb, None, fx=0.5, fy=0.5)
            # cv2.imshow("large_rgb_for_show", large_rgb_for_show)
            cv2.imshow("rgb_1", rgb_1)


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
