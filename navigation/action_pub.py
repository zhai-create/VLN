import numpy as np
from navigation.arguments import args


last_path_node = []

def normalize_angle_list(angle_list):
  """
    Transform the angle_list to [-180, 180]
    :param angle_list: List of arbitrary angles to be converted
    :return normalized_angles: angle list between [-180, 180]
  """
  normalized_angles = [angle % 360 for angle in angle_list]  # 使用列表解析进行取模运算
  normalized_angles = [angle if angle <180 else angle-360 for angle in normalized_angles]
  return np.array(normalized_angles)

def normalize_angle_to_180(angle):
  """
    Transform the angle to [-180, 180]
    :param angle: Arbitrary angles to be converted
    :return normalized_angle: angle between [-180, 180]
  """
  normalized_angle = angle % 360
  if normalized_angle > 180:
    normalized_angle = normalized_angle - 360
  return normalized_angle


def find_possible_direction(cur_dir, goal_dir):
  # 以角度制为单位,goal_dir在0～360度之间
  """
    Get the real goal turn-angle
    :param cur_dir: current turn-angle
    :param goal_dir: goal turn-angle
    :return candidate: Real goal turn-angle
  """
  candidate = np.linspace(cur_dir, cur_dir+350., num=36)
  candidate = normalize_angle_list(candidate) # 转换到-180--180度
  candidate = np.array([angle for angle in candidate if abs(normalize_angle_to_180(angle-goal_dir)) <= 60])
  diff = np.abs(candidate - goal_dir)
  idx = np.argsort(diff)
  candidate = candidate[idx] 
  return candidate


def calculate_angle(A, B, C):
  """
    Get the angle that the robot rotates around the next path point
    :param A: last path node
    :param B: next path node
    :param C: current robot's pos
    :return angle_deg: Angle that robot rotate around the next node path
  """
  # A：上一个path_node
  # B：下一个path_node
  # C：机器人位置
  # 将点坐标转换为numpy数组
  A = np.array(A)
  B = np.array(B)
  C = np.array(C)

  # 计算向量AB和BC
  BA = A - B
  BC = C - B

  # 计算向量AB和BC的点积
  dot_product = np.dot(BA, BC)

  # 计算向量AB和BC的模
  magnitude_BA = np.linalg.norm(BA)
  magnitude_BC = np.linalg.norm(BC)

  # 计算向量AB和BC的夹角（弧度）
  angle_rad = np.arccos(dot_product / (magnitude_BA * magnitude_BC))

  # 将弧度转换为角度
  angle_deg = np.degrees(angle_rad)

  # 判断角度是顺时针还是逆时针
  cross_product = np.cross(BA, BC)
  if cross_product < 0:
      angle_deg = 360 - angle_deg

  return angle_deg

def choose_action(action_category, state_flag, last_action, path, cur_rela_loc, cur_rela_dir):
  """
    Get the next action and update path based on path and current_pos (new version)
    :param action_category: action node_type
    :param state_flag: topo_planner state flag
    :param last_action: last selected action
    :param path: last updated rrt node_path
    :param cur_rela_loc: current robot relative loc
    :param cur_rela_dir: current robot relative dir
    :return path: updated node_path
    :return selected action
  """
  global last_path_node

  if(last_action!='l' and last_action!='r' and last_action!='f'):
    last_path_node = [cur_rela_loc[0], cur_rela_loc[1]]
    
  while len(path) > 0:
    d = np.sqrt((cur_rela_loc[0] - path[0][0])**2 + (cur_rela_loc[1] - path[0][1])**2) #格
    now_robot_position = [cur_rela_loc[0], cur_rela_loc[1]]
    res_angle = calculate_angle(last_path_node, [path[0][0], path[0][1]], now_robot_position)
    
    if(res_angle>180):
      res_angle = 360-res_angle

    if (action_category=="frontier_node") and (state_flag=="finish"): # 如果是frontier并且以frontier为最终的goal
      temp_d = np.sqrt((cur_rela_loc[0] - path[-1][0])**2 + (cur_rela_loc[1] - path[-1][1])**2) #格
      if temp_d<args.frontier_dis_thre:
        path = []
        break

    if d <= args.dis_thre or res_angle>args.angle_thre:
      last_path_node = [path[0][0], path[0][1]]
      path.pop(0)
      break
    else:
      break

  if len(path) == 0:
    return path, "suc"
  else:
    target = path[0]
    final_goal = path[len(path)-1]
    d = np.sqrt((cur_rela_loc[0] - target[0])**2 + (cur_rela_loc[1] - target[1])**2) #格
    d_to_final = np.sqrt((cur_rela_loc[0] - final_goal[0])**2 + (cur_rela_loc[1] - final_goal[1])**2) #格

    goal_dir = np.arctan2(-target[0]+cur_rela_loc[0], target[1]-cur_rela_loc[1])*180/np.pi
    
    try:
      possible_direction = (find_possible_direction(cur_rela_dir, goal_dir))[0] #在当前节点的固定坐标系下
      diff = normalize_angle_to_180(possible_direction-cur_rela_dir)
    except:
      diff = 0

    if abs(diff) < args.turn_angle:
      return path, "f"
    elif diff >= args.turn_angle:
      return path, "l"
    else:
      return path, "r"


def choose_action_origin(path, cur_rela_loc, cur_rela_dir):
  """
    Get the next action and update path based on path and current_pos (old version)
    :param path: last updated rrt node_path
    :param cur_rela_loc: current robot relative loc
    :param cur_rela_dir: current robot relative dir
    :return path: updated node_path
    :return selected action
  """
  #坐标系：
  #x：往下，y：往右
  while len(path) > 0:
    d = np.sqrt((cur_rela_loc[0] - path[0][0])**2 + (cur_rela_loc[1] - path[0][1])**2) #格
    print("**************")
    print("cur_rela_loc:", cur_rela_loc)
    print("path[0]:", path[0])
    print("current_path0_dis:", d)
    print("**************")
    
    if len(path) == 1:
      if d <=5:
        path.pop(0)
      break

    elif d <= 1.3:
      path.pop(0)
      continue

    elif d <= 2.5:
      # decide = forward_or_not()
      path.pop(0)
      break

    else:
      break


  if len(path) == 0:
    return path, "suc"
  else:
    target = path[0]
    final_goal = path[len(path)-1]
    d = np.sqrt((cur_rela_loc[0] - target[0])**2 + (cur_rela_loc[1] - target[1])**2) #格
    d_to_final = np.sqrt((cur_rela_loc[0] - final_goal[0])**2 + (cur_rela_loc[1] - final_goal[1])**2) #格
    print("ACTION PUB dis_to_next_point(meter): ", d/10)
    print("ACTION PUB dis_to_goal(meter): ", d_to_final/10)
    print("points left: ", len(path))

    goal_dir = np.arctan2(-target[0]+cur_rela_loc[0], target[1]-cur_rela_loc[1])*180/np.pi
    print("ACTION PUB goal_dir: ", goal_dir)
    possible_direction = (find_possible_direction(cur_rela_dir, goal_dir))[0] #在当前节点的固定坐标系下
    print("ACTION PUB possible_direction: ", possible_direction)
    diff = normalize_angle_to_180(possible_direction-cur_rela_dir)
    print("ACTION PUB diff: ", diff)
    if abs(diff) < args.turn_angle:
      return path, "f"
    elif diff >= args.turn_angle:
      return path, "l"
    else:
      return path, "r"