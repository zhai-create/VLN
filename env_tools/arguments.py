import argparse


def get_args():
    parser = argparse.ArgumentParser(description="train-and-test-revelent")

    parser.add_argument("--task_stage", type=str, default="val")

    # test revelent
    parser.add_argument("--graph_train", type=bool, default=False)
    parser.add_argument("--root", type=str, default="/home/zhaishichao/Data/VLN_master")
    parser.add_argument("--model_file_name", type=str, default="")
    parser.add_argument("--graph_pre_model", type=int, default=-1)
    parser.add_argument("--logger_file_name", type=str, default="")
    parser.add_argument("--graph_episode_num", type=int, default=-1)

    # train revelent
    parser.add_argument("--graph_model_save_frequency", type=int, default=1)
    parser.add_argument("--graph_batch_size", type=int, default=-1)
    parser.add_argument("--graph_episode_length", type=int, default=-1) # 每个episode中rl的最大次数
    parser.add_argument("--graph_iter_per_step", type=int, default=-1)

    # episodes revelent
    # parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--success_distance", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--object_ls", type=list, default=["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"])

    parser.add_argument("--is_auto", type=bool, default=True)
    

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()