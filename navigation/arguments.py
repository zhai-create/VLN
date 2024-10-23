import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="navigation-revelent")

    # General Arguments
    parser.add_argument("--inflation_distance", type=float, default=0.5)
    parser.add_argument("--large_dis_thre", type=int, default=10)
    parser.add_argument("--small_dis_thre", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=2)

    # RRT revelent
    parser.add_argument("--obstable_threshold_unout", type=float, default=0.51)
    parser.add_argument("--obstable_threshold_unin", type=float, default=0.49)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--min_step_size", type=float, default=1)
    parser.add_argument("--max_step_size", type=float, default=3)
    parser.add_argument("--goal_step_size", type=float, default=2.5)
    parser.add_argument("--goal_threshold", type=float, default=5.0)
    parser.add_argument("--goal_sample_rate", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--obstacle_free_step_size", type=float, default=1.0)
    parser.add_argument("--obstable_distance", type=float, default=1.0)
    parser.add_argument("--free_map_delta", type=int, default=1)
    parser.add_argument("--min_dis_delta", type=float, default=1e-10)
    parser.add_argument("--new_item_factor", type=float, default=0.025)

    # ASTAR revelent
    parser.add_argument("--astar_grid_delta", type=int, default=7)
    parser.add_argument("--cost_factor", type=float, default=10)
    parser.add_argument("--time_thre", type=float, default=180)

    # action_pub revelent
    parser.add_argument("--frontier_dis_thre", type=float, default=5)
    parser.add_argument("--dis_thre", type=float, default=1)
    parser.add_argument("--angle_thre", type=float, default=50)
    parser.add_argument("--max_rotate_cnt", type=int, default=4)
    parser.add_argument("--len_action_buffer_thre", type=int, default=4)

    parser.add_argument("--forward_dis", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=float, default=30)
    # parser.add_argument("--turn_angle", type=float, default=10)

    parser.add_argument("--path_block_ls_thre", type=int, default=4)
    parser.add_argument("--path_block_meter_thre", type=float, default=0.5)

    parser.add_argument("--FINISH", type=str, default="f")
    parser.add_argument("--OTHER", type=str, default="o")
    parser.add_argument("--GO_ON", type=str, default="a")

    parser.add_argument("--is_choose_action", type=bool, default=False)

    
    
    

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()