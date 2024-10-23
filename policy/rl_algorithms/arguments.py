import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="rl-revelent")

    parser.add_argument("--is_see_grid_delta", type=int, default=5)
    parser.add_argument("--score_top_k", type=int, default=50)

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr_tune", type=float, default=1.5e-3)
    parser.add_argument("--alpha_init", type=float, default=1.0)

    # feature_dim
    parser.add_argument('--graph_node_feature_dim', type=int, default=2,
        help="the dimension of graph node")
    parser.add_argument('--graph_edge_feature_dim', type=int, default=3,
        help="the dimension of graph edge")
    parser.add_argument('--graph_embedding_dim', type=int, default=64,
        help="the dimension of graph embedding")

    # padding
    parser.add_argument('--graph_num_action_padding', type=int, default=500,
        help="the padding size used in potential action nodes  ")
    parser.add_argument('--graph_num_graph_padding', type=int, default=-1,
        help="the padding size used in graph nodes  ")

    # graph_type
    parser.add_argument('--graph_using_pyg', action='store_true', default=True,
        help="if toggled, use python geometric library for graph data")
    parser.add_argument('--graph_encoder', type=str, default='GAT',
        help="the type of graph encoder :  ['GCN', 'GAT', 'Transformer']  ")
    parser.add_argument('--graph_sac_greedy', action='store_true', default=True,
        help="if toggled, use start greedy sac in graph task")
    parser.add_argument('--buffer_size', type=int, default=5e5,
        help="the buffer size in training")
    parser.add_argument('--lr_scheduler_interval', type=int, default=1e2,
        help="the interval of the learning rate scheduler step in training")
    parser.add_argument('--graph_action_space', type=str, default='adjacent',
        help="the action space of the graph task  :  ['adjacent', 'frontier']  ")

    # graph task setting
    parser.add_argument('--graph_task', type=str, default='object_goal_navigation',
        help="the type of the graph task  :  ['object_goal', 'image_goal', 'VLN']  ")


    # lr
    parser.add_argument('--graph_lr_actor', type=float, default=1e-4,
        help="the learning rate of actor network") 
    parser.add_argument('--graph_lr_critic', type=float, default=1e-4,
        help="the learning rate of critic network")
    parser.add_argument("--random_exploration_length", type=int, default=-1)

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()