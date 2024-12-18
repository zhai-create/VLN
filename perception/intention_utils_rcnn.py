import torch
import copy
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from perception.arguments import args, coco_categories_mapping
from perception.tools import sam_show_mask, depth_estimation, depth_estimation_laser

from env_tools.arguments import args as env_args


def object_detect(rgb_image_ls, depth, object_text):
    """
    Get the candidate intention nodes, which need the pos transformer and the selection
    :param rgb_image_ls: [front_img, left_img, behind_img, right_img]
    :param depth: depth after fixed
    :param object_text: object-goal
    :return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}
    """
    detect_res_pos_dict = {}

    img_info_inputs = []
    # 初始化img信息
    for original_image in rgb_image_ls:
        original_image = original_image[:, :, ::-1]
        img_height, img_width = original_image.shape[:2]
        original_image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
        img_instance = {"image": original_image, "height": img_height, "width": img_width}
        img_info_inputs.append(img_instance)

    with torch.no_grad():
        detections_ls = rcnn_model(img_info_inputs)

    for index in range(len(rgb_image_ls)): # 遍历每一张图片
    # for index in range(1): # 遍历每一张图片
        temp_outputs = detections_ls[index]
        temp_masks = temp_outputs["instances"].pred_masks
        temp_boxes = temp_outputs["instances"].pred_boxes
        temp_boxes = temp_boxes.tensor.cpu().numpy()
        temp_pre_labels = temp_outputs["instances"].pred_classes
        temp_pre_scores = temp_outputs["instances"].scores

        for temp_index in range(temp_boxes.shape[0]): # 遍历每一个图像实例
            if(int(temp_pre_labels[temp_index].item())==coco_categories_mapping[object_text]):
                new_mask = temp_masks.cpu().numpy()[temp_index]
                
                # show
                # image0 = rgb_image_ls[index].copy()
                # cv2.rectangle(image0, (int(temp_boxes[temp_index][0]), int(temp_boxes[temp_index][1])), (int(temp_boxes[temp_index][2]), int(temp_boxes[temp_index][3])), (0, 255, 0), 2)
                # sam_show_res = sam_show_mask(new_mask, image0)
                # cv2.imwrite("sam_show_res_{}_{}.jpg".format(index, temp_index), sam_show_res)
                
                if(env_args.is_one_rgb==True): # 是使用一张rgb
                    new_add_half_width = int(((new_mask.shape[1]*90/79)-new_mask.shape[1])/2)
                    new_add_false_matrix = np.zeros((new_mask.shape[0], new_add_half_width), dtype=bool)
                    new_mask = np.hstack((new_add_false_matrix, new_mask, new_add_false_matrix))
                
                
                false_matrix = np.zeros(new_mask.shape, dtype=bool)
                if((index+1)==1):
                    large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], false_matrix, new_mask, false_matrix, false_matrix[:, :int(false_matrix.shape[1]//2)]))
                elif((index+1)==2):
                    large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], new_mask, false_matrix, false_matrix, false_matrix[:, :int(false_matrix.shape[1]//2)]))
                elif((index+1)==4):
                    large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], false_matrix, false_matrix, new_mask, false_matrix[:, :int(false_matrix.shape[1]//2)]))
                elif((index+1)==3):
                    large_mask = np.hstack((new_mask[:, int(false_matrix.shape[1]//2):], false_matrix, false_matrix, false_matrix, new_mask[:, :int(false_matrix.shape[1]//2)]))



                if(args.is_depth_estimation_laser==True):
                    res_depth_2d_cx, res_depth_2d_cy = depth_estimation_laser(large_mask, depth)
                else:
                    res_depth_2d_cx, res_depth_2d_cy = depth_estimation(large_mask, depth) # 相对于机器人的位姿
                
                if(res_depth_2d_cx is None):
                    continue
                
                if(temp_pre_scores[temp_index].item() not in detect_res_pos_dict):
                    detect_res_pos_dict[temp_pre_scores[temp_index].item()] = [[res_depth_2d_cx, res_depth_2d_cy]]
                else:
                    detect_res_pos_dict[temp_pre_scores[temp_index].item()].append([res_depth_2d_cx, res_depth_2d_cy])
    return detect_res_pos_dict


"""
Perpection Model Init, detect & mask
"""

# loading config
# args.mask_rcnn_thre = 0.6

rcnn_cfg = get_cfg()
rcnn_cfg.merge_from_file(args.rcnn_yaml_path)
rcnn_cfg.merge_from_list(["MODEL.WEIGHTS", args.rcnn_weight_path])
rcnn_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.mask_rcnn_thre
rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.mask_rcnn_thre
rcnn_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
    args.mask_rcnn_thre
)
rcnn_cfg.MODEL.DEVICE = args.model_device  
rcnn_cfg.freeze()

# init rcnn
rcnn_cfg = rcnn_cfg.clone()
rcnn_model = build_model(rcnn_cfg)
rcnn_model.eval()
checkpointer = DetectionCheckpointer(rcnn_model)
checkpointer.load(rcnn_cfg.MODEL.WEIGHTS)
print('Mask-rcnn initialize success!')

print("\n\n\n\n\n")
print("args.mask_rcnn_thre:", args.mask_rcnn_thre)


