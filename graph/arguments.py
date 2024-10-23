import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="graph-revelent")

    # General Arguments
    parser.add_argument("--resolution", type=float, default=0.1, help="the map resolution")
    parser.add_argument("--free_val", type=float, default=0.3, help="free region")
    parser.add_argument("--unknown_val", type=float, default=0.5, help="unknown region")
    parser.add_argument("--obstacle_val", type=float, default=0.7, help="obstacle region")

    parser.add_argument("--obstacle_dis", type=float, default=0.01, help="if it is a obstacle region")
    parser.add_argument("--clip_lower", type=float, default=-3, help="the lower thre of map clip")
    parser.add_argument("--clip_upper", type=float, default=3, help="the upper thre of map clip")
    parser.add_argument("--ghost_map_thre", type=float, default=0.69)
    parser.add_argument("--ghost_map_g_val", type=float, default=0.5)
    parser.add_argument("--ghost_map_delta", type=float, default=0.1)
    parser.add_argument("--ghost_diff_thre", type=float, default=0.19)
    parser.add_argument("--thre_for_delete", type=int, default=3)
    parser.add_argument("--thre_for_detect", type=int, default=4)

    parser.add_argument("--thre_for_blacklist_delete", type=float, default=1.0)
    parser.add_argument("--d_gap", type=float, default=1.1)
    parser.add_argument("--min_around_thre", type=float, default=0.31)
    parser.add_argument("--diff_upper", type=float, default=0.1)
    parser.add_argument("--third_check_thre", type=float, default=0.16)
    parser.add_argument("--init_predict_ghost_thre1", type=float, default=1.2)

    parser.add_argument("--clear_fake_lower", type=int, default=-3)
    parser.add_argument("--clear_fake_upper", type=int, default=4)
    parser.add_argument("--grid_delta", type=int, default=10)
    

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()