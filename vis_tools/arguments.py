import argparse

def get_args():
    parser = argparse.ArgumentParser(description="vis-revelent")

    # General Arguments
    parser.add_argument("--font_width", type=int, default=1)
    parser.add_argument("--font_size", type=float, default=0.7)
    
    parser.add_argument("--font_pos1", type=tuple, default=(10, 15))
    parser.add_argument("--font_pos2", type=tuple, default=(10, 35))
    parser.add_argument("--font_pos3", type=tuple, default=(10, 55))
    parser.add_argument("--new_top_left", type=tuple, default=(0, 0))
    parser.add_argument("--new_right_bottom", type=tuple, default=(240, 60))

    # parser.add_argument("--pre_path", type=str, default="vis_result_0107_onlygt_plan")
    # parser.add_argument("--pre_path", type=str, default="vis_result_0107_onlygt")
    
    # parser.add_argument("--pre_path", type=str, default="vis_result_0107_onlygt_fakeintention_no_filter")
    # parser.add_argument("--pre_path", type=str, default="vis_result_0107_onlygt_ctr")
    # parser.add_argument("--pre_path", type=str, default="vis_result_0107_onlygt_ctr_plan_fakeintention")

    # parser.add_argument("--pre_path", type=str, default="vis_result_one_rgb_rotate")
    parser.add_argument("--pre_path", type=str, default="vis_result_frontier_cluster_gt")
    # parser.add_argument("--pre_path", type=str, default="vis_result_many_intention")

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()