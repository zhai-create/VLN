import torch
import copy
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from perception.arguments import args, coco_categories_mapping
from perception.tools import sam_show_mask, depth_estimation, depth_estimation_laser, depth_estimation_laser_pinhole_to_panorama, depth_estimation_object_loc

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

    return temp_masks, temp_boxes, temp_pre_scores


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


test_img1 = cv2.imread("/home/zhaishichao/Data/VLN/save_rgb/1741598048.3027678.jpg")
test_img2 = cv2.imread("/home/zhaishichao/Data/VLN/save_rgb/1741598049.3499246.jpg")



masks_1, boxes_1, pre_scores_1 = object_detect([test_img1], None, "plant")
masks_2, boxes_2, pre_scores_2 = object_detect([test_img2], None, "plant")


cv2.rectangle(test_img1, (int(boxes_1[0][0]), int(boxes_1[0][1])), (int(boxes_1[0][2]), int(boxes_1[0][3])), (0, 255, 0), 2)
cv2.imwrite("test_img1.jpg", test_img1)

cv2.rectangle(test_img2, (int(boxes_2[0][0]), int(boxes_2[0][1])), (int(boxes_2[0][2]), int(boxes_2[0][3])), (0, 255, 0), 2)
cv2.imwrite("test_img2.jpg", test_img1)

res_box_1 = boxes_1[0].reshape(1, 4)
res_box_2 = boxes_2[0].reshape(1, 4)

print(boxes_1)

print(np.array([boxes_1[0], boxes_2[0]]))

# np.save("res_box_1.npy", res_box_1)
# np.save("res_box_2.npy", res_box_2)


# res_dict_1 = {"res_box_1": res_box_1, }


