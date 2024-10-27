import numpy as np
from navigation.habitat_action import HabitatAction
from env_tools.arguments import args

class Evaluate:
    success_num = 0 # sr
    spl_per_episode = 0
    spl_ls = []
    spl_mean = 0
    
    ne_per_episode = 0
    ne_ls = []
    ne_mean = 0

    state_dict = {"achieved": 1, "exceed": -1, "Failed_Plan": -2, "empty": -3, "block": -4}

    @staticmethod
    def evaluate(writer, achieved_result, habitat_env, action_node, index_in_episodes):
        # 1. 成功到达
        # 0. 到达错误的label goal
        # -1. 超过最大步长
        # -2. rrt规划失败
        # -3. action space为空
        # -4. 卡住（只记录intention node的卡住）
        
        habitat_metric = habitat_env.get_metrics()
        print("habitat_metric:", habitat_metric)
        if(achieved_result=="exceed" or achieved_result=="empty"):
            if(habitat_metric['success']>0):
                Evaluate.success_num += 1
            Evaluate.spl_per_episode = habitat_metric['spl']
            Evaluate.spl_ls.append(Evaluate.spl_per_episode)
            Evaluate.spl_mean = np.mean(Evaluate.spl_ls)

            Evaluate.ne_per_episode = habitat_metric['distance_to_goal']
            Evaluate.ne_ls.append(Evaluate.ne_per_episode)
            Evaluate.ne_mean = np.mean(Evaluate.ne_ls)
            
            writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
            writer.add_scalar('Result/walk_path_meter', HabitatAction.walk_path_meter, index_in_episodes+1)
            writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
            writer.add_scalar('Result/success_num', Evaluate.success_num, index_in_episodes+1)
            writer.add_scalar('Result/spl_per_episode', Evaluate.spl_per_episode, index_in_episodes+1)
            writer.add_scalar('Result/spl_mean', Evaluate.spl_mean, index_in_episodes+1)
            writer.add_scalar('Result/ne_per_episode', Evaluate.ne_per_episode, index_in_episodes+1)
            writer.add_scalar('Result/ne_mean', Evaluate.ne_mean, index_in_episodes+1)
            return "episode_stop" # 结束当前episode, 开始下一个episode
            
        elif(achieved_result=="achieved" or achieved_result=="block" or achieved_result=="Failed_Plan"):
            if(action_node.node_type=="frontier_node"):
                return "next_rl_step" # 继续选择下一个action

            elif(action_node.node_type=="intention_node"):
                if(habitat_metric['success']>0):
                    Evaluate.success_num += 1
                Evaluate.spl_per_episode = habitat_metric['spl']
                Evaluate.spl_ls.append(Evaluate.spl_per_episode)
                Evaluate.spl_mean = np.mean(Evaluate.spl_ls)

                Evaluate.ne_per_episode = habitat_metric['distance_to_goal']
                Evaluate.ne_ls.append(Evaluate.ne_per_episode)
                Evaluate.ne_mean = np.mean(Evaluate.ne_ls)
                
                writer.add_scalar('Result/episode_state', Evaluate.state_dict[achieved_result], index_in_episodes+1)
                writer.add_scalar('Result/walk_path_meter', HabitatAction.walk_path_meter, index_in_episodes+1)
                writer.add_scalar('Result/this_episode_short_dis', HabitatAction.this_episode_short_dis, index_in_episodes+1)
                writer.add_scalar('Result/success_num', Evaluate.success_num, index_in_episodes+1)
                writer.add_scalar('Result/spl_per_episode', Evaluate.spl_per_episode, index_in_episodes+1)
                writer.add_scalar('Result/spl_mean', Evaluate.spl_mean, index_in_episodes+1)
                writer.add_scalar('Result/ne_per_episode', Evaluate.ne_per_episode, index_in_episodes+1)
                writer.add_scalar('Result/ne_mean', Evaluate.ne_mean, index_in_episodes+1)
                return "episode_stop" # 结束当前episode, 开始下一个episode
                