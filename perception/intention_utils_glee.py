import cv2
import torch
import numpy as np
import torchvision

import torch.nn.functional as F

from detectron2.config import get_cfg
from perception.glee_tools import add_deeplab_config, add_glee_config
from perception.arguments import args
from perception.object_list import categories as CATEGORIES

from dependencies.GLEE.glee.models.glee_model import GLEE_Model
from habitat_sim.utils.common import d3_40_colors_rgb

CATEGORIES = [cat['name'].lower() for cat in CATEGORIES]

def visualize_segmentation(image,classes,masks):
    copy_image = image.copy()
    label_classes = np.unique(classes)
    for cls,mask in zip(classes,masks):
        if len(np.unique(mask)) != 2: continue
        copy_image[np.where(mask == 1)] = d3_40_colors_rgb[label_classes.tolist().index(cls)]
        x, y = int(np.mean(np.where(mask)[1])), int(np.mean(np.where(mask)[0])) 
        cv2.putText(copy_image, str(cls), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    ret_image = cv2.addWeighted(image,0.2,copy_image,0.8,0)
    return ret_image

def object_detect(rgb_image_ls, depth=None, object_text=None):
    """
    Get the candidate intention nodes, which need the pos transformer and the selection
    :param rgb_image_ls: [front_img, left_img, behind_img, right_img]
    :param depth: depth after fixed
    :param object_text: object-goal
    :return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}
    """

    object_goal_label_mapper = {}


    detect_res_pos_dict = {}
    for index in range(len(rgb_image_ls)): # 遍历每一张图片
        # rgb_image_ls[index] = cv2.cvtColor(rgb_image_ls[index], cv2.COLOR_BGR2RGB)
        pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(rgb_image_ls[index], glee_model, threshold_select=confidence_threshold, device=args.model_device)
        
        # pred_bboxes: (None, 4)
        # pred_masks: (None, 512, 512)
        # pred_class: ['bed' 'curtains' 'lamp' 'mirror' 'vase' 'blinds' 'photo_frame' 'table', 'drawer' 'table' 'fabric']
        # pred_confidence: tensor([0.6429, 0.5060, 0.4873, 0.4557, 0.3881, 0.3687, 0.3533, 0.3354, 0.3238, 0.3209, 0.2534], device='cuda:1')
    
        try:
            mask_area = np.array([mask.sum() for mask in pred_masks])
            # visualization = visualize_segmentation(rgb_image_ls[index],pred_class[(mask_area>area_threshold)],pred_masks[(mask_area>area_threshold)])
            pred_class_ls = pred_class[(mask_area>area_threshold)]
            pred_masks_ls = pred_masks[(mask_area>area_threshold)]
            pred_confidence_ls = pred_confidence[(mask_area>area_threshold)]


            
        
            
        except:
            return detect_res_pos_dict



def initialize_glee(glee_config="./dependencies/GLEE/configs/SwinL.yaml",
                    glee_checkpoint="./dependencies/GLEE/GLEE_SwinL_Scaleup10m.pth",
                    device=args.model_device):
    cfg_swin = get_cfg()
    add_deeplab_config(cfg_swin)
    add_glee_config(cfg_swin)
    conf_files_swin = glee_config
    checkpoints_swin = torch.load(glee_checkpoint) 
    cfg_swin.merge_from_file(conf_files_swin)
    GLEEmodel_swin = GLEE_Model(cfg_swin, None, device, None, True).to(device)
    GLEEmodel_swin.load_state_dict(checkpoints_swin, strict=False)
    GLEEmodel_swin.eval()
    return GLEEmodel_swin

def glee_segmentation(img, 
                      GLEEmodel, 
                      custom_category=CATEGORIES,
                      num_inst_select=15,
                      threshold_select=0.2,
                      device=args.model_device):        
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    ori_image = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
    ori_image = normalizer(ori_image.to(device))[None,]
    _,_, ori_height, ori_width = ori_image.shape
    resizer = torchvision.transforms.Resize(800)
    resize_image = resizer(ori_image)
    image_size = torch.as_tensor((resize_image.shape[-2],resize_image.shape[-1]))
    re_size = resize_image.shape[-2:]
    stride = 32
    # the last two dims are H,W, both subject to divisibility requirement
    padding_size = ((image_size + (stride - 1)).div(stride, rounding_mode="floor") * stride).tolist()
    infer_image = torch.zeros(1,3,padding_size[0],padding_size[1]).to(resize_image)
    infer_image[0,:,:image_size[0],:image_size[1]] = resize_image
    batch_category_name = custom_category
    prompt_list = []
    with torch.no_grad():
        (outputs,_) = GLEEmodel(infer_image, prompt_list, task="coco", batch_name_list=batch_category_name, is_train=False)
    topK_instance = max(num_inst_select,1)
    bbox_pred = outputs['pred_boxes'][0]
    bbox_pred[:,0],bbox_pred[:,2] = bbox_pred[:,0] * img.shape[1] - bbox_pred[:,2] * img.shape[1] * 0.5, bbox_pred[:,0] * img.shape[1] + bbox_pred[:,2] * img.shape[1] * 0.5
    bbox_pred[:,1],bbox_pred[:,3] = bbox_pred[:,1] * img.shape[0] - bbox_pred[:,3] * img.shape[0] * 0.5, bbox_pred[:,1] * img.shape[0] + bbox_pred[:,3] * img.shape[0] * 0.5
    mask_pred = outputs['pred_masks'][0]
    mask_cls = outputs['pred_logits'][0]
    scores = mask_cls.sigmoid().max(-1)[0]
    scores_per_image, topk_indices = scores.topk(topK_instance, sorted=True)
    valid = scores_per_image>threshold_select
    topk_indices = topk_indices[valid]
    scores_per_image = scores_per_image[valid]
    pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
    if len(pred_class) == 0: 
        return [], [], [], []
    mask_pred = mask_pred[topk_indices]
    bbox_pred = bbox_pred[topk_indices].cpu().numpy()
    pred_masks = F.interpolate( mask_pred[None,], size=(padding_size[0], padding_size[1]), mode="bilinear", align_corners=False)
    pred_masks = pred_masks[:,:,:re_size[0],:re_size[1]]
    pred_masks = F.interpolate( pred_masks, size=(ori_height,ori_width), mode="bilinear", align_corners=False)
    pred_masks = (pred_masks>0).detach().cpu().numpy()[0]
    return bbox_pred, pred_masks, np.array(batch_category_name)[pred_class], scores_per_image

confidence_threshold = 0.25
area_threshold=2500
glee_model = initialize_glee()


rgb1 = cv2.imread("okok1.jpg")
classes,masks,confidences,visualization = object_detect([rgb1], depth=None, object_text=None)




