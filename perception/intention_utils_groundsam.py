import cv2
import copy
import torch
import numpy as np
from PIL import Image

from perception.arguments import args
from perception.tools import depth_estimation_object_loc, sam_show_mask

from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)


def cxcywh_to_xyxy(boxes):
    """
    Args:
        boxes: (N, 4) 或 (4,), 格式为 (cx, cy, w, h)
    Returns:
        xyxy_boxes: (N, 4) 或 (4,), 格式为 (x1, y1, x2, y2)
    """
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)



def object_detect_grounding_sam(rgb_image_ls, depth, object_text):
    """
    使用GroundingSAM进行目标检测的版本
    :param rgb_image_ls: [front_img, left_img, behind_img, right_img]
    :param depth: 深度图
    :param object_text: 目标文本描述
    :return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}
    """
    detect_res_pos_dict = {}

    for index, original_image in enumerate(rgb_image_ls):  # 遍历每张图片
        # 转换图像格式
        image_source = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_source)
        image_transformed, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=grounding_dino_model,
            image=image_transformed,
            caption=object_text.lower() + ".",  # 文本提示需要特定格式
            # box_threshold=0.35,
            # text_threshold=0.25
            box_threshold=0.6,
            text_threshold=0.6
        )
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = cxcywh_to_xyxy(boxes)

        if len(boxes) == 0:
            return detect_res_pos_dict

        # 准备SAM输入
        sam_image = copy.deepcopy(rgb_image_ls[index])
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(sam_image)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes.to(args.model_device),
            sam_image.shape[:2]
        ).to(args.model_device)

        # SAM生成掩码
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # 处理检测结果
        for mask_idx in range(len(masks)):
            score = logits[mask_idx].item()
            mask = masks[mask_idx].cpu().numpy()[0]  # 获取第一个掩码

            # # show
            # image0 = rgb_image_ls[index].copy()
            # cv2.rectangle(image0, (int(boxes[mask_idx][0]), int(boxes[mask_idx][1])), (int(boxes[mask_idx][2]), int(boxes[mask_idx][3])), (0, 255, 0), 2)
            # sam_show_res = sam_show_mask(mask, image0)
            # cv2.imwrite("groundsam_show_res_{}.jpg".format(mask_idx+10), image0)
            # print("score:", score)
            # print("boxes:", boxes)
            # print("transformed_boxes:", transformed_boxes)
            
            
            # 深度估计
            res_depth_2d_cx, res_depth_2d_cy = depth_estimation_object_loc(mask, depth)

            if (res_depth_2d_cx is None) or ((res_depth_2d_cx**2 + res_depth_2d_cy**2)**0.5 < 0.75):
                continue

            # 保存结果
            if score not in detect_res_pos_dict:
                detect_res_pos_dict[score] = [[res_depth_2d_cx, res_depth_2d_cy]]
            else:
                detect_res_pos_dict[score].append([res_depth_2d_cx, res_depth_2d_cy])
    return detect_res_pos_dict



grounding_dino_model = load_model('./dependencies/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', './dependencies/models/groundingdino_swint_ogc.pth', device=args.model_device)
sam_predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint='./dependencies/models/sam_vit_h_4b8939.pth').to(args.model_device))
transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
print('groundsam initialize success!')


# test_img = cv2.imread("okok1.jpg")
# object_detect_grounding_sam([test_img], None, "bed")