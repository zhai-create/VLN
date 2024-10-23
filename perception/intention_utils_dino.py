import torch
import copy
import cv2
import numpy as np

from segment_anything_hq import SamPredictor, sam_model_registry
from sentence_transformers import SentenceTransformer
from dependencies.GroundingDINO.groundingdino.util.inference import Model

from perception.arguments import args
from perception.tools import sam_show_mask, depth_estimation


def object_detect(rgb_image_ls, depth, object_text):
    """
    Get the candidate intention nodes, which need the pos transformer and the selection
    :param rgb_image_ls: [front_img, left_img, behind_img, right_img]
    :param depth: depth after fixed
    :param object_text: object-goal
    :return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}
    """
    detect_res_pos_dict = {}
    detections_ls = grounding_dino_model.new_predict_with_classes(
        raw_image_ls=rgb_image_ls,
        classes=[object_text],
        box_threshold=args.BOX_THRESHOLD,
        text_threshold=args.BOX_THRESHOLD
    )

    for index in range(len(rgb_image_ls)): # 遍历每一张图片
        temp_box_ls = detections_ls[index].xyxy
        temp_class_id_ls = detections_ls[index].class_id
        temp_confidence_ls = detections_ls[index].confidence
        if(len(temp_box_ls)>0):
            sam_image = copy.deepcopy(rgb_image_ls[index])
            sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(sam_image)

            input_box = torch.tensor([[int(temp_box_ls[temp_box_ls_index][0]/args.factor_0), int(temp_box_ls[temp_box_ls_index][1]/args.factor_1), int(temp_box_ls[temp_box_ls_index][2]/args.factor_0), int(temp_box_ls[temp_box_ls_index][3]/args.factor_1)]  for temp_box_ls_index in range(len(temp_class_id_ls))], device=sam_predictor.device)
            transformed_box = sam_predictor.transform.apply_boxes_torch(input_box, sam_image.shape[:2])

            masks, scores, logits = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = object_text,
                boxes = transformed_box,
                multimask_output = False,
            )

        else:
            continue
        # =========== new add multi sam end ==============
        for temp_index in range(len(temp_class_id_ls)): # 遍历同一张图片检测出来的每一个result
            if(temp_confidence_ls[temp_index]<args.CONFIDENCE_TRESHOLE or temp_class_id_ls[temp_index]==None):
                continue
            
            new_mask = masks.cpu().numpy()[temp_index][0]

            # show
            # image0 = rgb_image_ls[index].copy()
            # cv2.rectangle(image0, (int(temp_box_ls[temp_index][0]/args.factor_0), int(temp_box_ls[temp_index][1]/args.factor_1)), (int(temp_box_ls[temp_index][2]/args.factor_0), int(temp_box_ls[temp_index][3]/args.factor_1)), (0, 255, 0), 2)
            # sam_show_res = sam_show_mask(new_mask, image0)
            # cv2.imwrite("sam_show_res_{}.jpg".format(temp_index+1), sam_show_res)

            false_matrix = np.zeros(new_mask.shape, dtype=bool)
            if((index+1)==1):
                large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], false_matrix, new_mask, false_matrix, false_matrix[:, :int(false_matrix.shape[1]//2)]))
            elif((index+1)==2):
                large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], new_mask, false_matrix, false_matrix, false_matrix[:, :int(false_matrix.shape[1]//2)]))
            elif((index+1)==4):
                large_mask = np.hstack((false_matrix[:, int(false_matrix.shape[1]//2):], false_matrix, false_matrix, new_mask, false_matrix[:, :int(false_matrix.shape[1]//2)]))
            elif((index+1)==3):
                large_mask = np.hstack((new_mask[:, int(false_matrix.shape[1]//2):], false_matrix, false_matrix, false_matrix, new_mask[:, :int(false_matrix.shape[1]//2)]))

            res_depth_2d_cx, res_depth_2d_cy = depth_estimation(large_mask, depth) # 相对于机器人的位姿
            if(temp_confidence_ls[temp_index] not in detect_res_pos_dict):
                detect_res_pos_dict[temp_confidence_ls[temp_index]] = [[res_depth_2d_cx, res_depth_2d_cy]]
            else:
                detect_res_pos_dict[temp_confidence_ls[temp_index]].append([res_depth_2d_cx, res_depth_2d_cy])
    return detect_res_pos_dict


"""
Perpection Model Init, detect & mask
"""



# init dino
grounding_dino_model = Model(model_config_path=args.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=args.GROUNDING_DINO_CHECKPOINT_PATH)
# init sam
sam_checkpoint = args.sam_path 
sam_version = args.sam_type
sam_predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(args.model_device))
print('SAM initialize success!')


