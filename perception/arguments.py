import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="perception-revelent")

    # General Arguments
    parser.add_argument("--camera_height", type=float, default=0.88)
    parser.add_argument("--height_thre", type=float, default=0.2, help="height threshold for judging if it is a obstacle")
    parser.add_argument("--depth_height", type=int, default=241)
    parser.add_argument("--depth_width", type=int, default=481)
    parser.add_argument("--depth_scale", type=int, default=10)
    parser.add_argument("--filter_thre", type=float, default=0.25)
    parser.add_argument("--index_ratio", type=int, default=1, help="get the 1/index_ratio of the original laser_2d_filtered")
    parser.add_argument("--angle_depth", type=float, default=35.264389682754654, help="angle for pre_depth")

    parser.add_argument("--sam_path", type=str, default='/home/zhaishichao/Data/Grounded_Segment_Anything/sam_hq_vit_h.pth')
    parser.add_argument("--sam_type", type=str, default='vit_h')
    parser.add_argument("--model_device", type=str, default='cuda:0')
    parser.add_argument("--text_to_text_path", type=str, default='/home/zhaishichao/Data/Grounded_Segment_Anything/all-MiniLM-L6-v2')
    parser.add_argument("--GROUNDING_DINO_CONFIG_PATH", type=str, default="/home/zhaishichao/Data/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--GROUNDING_DINO_CHECKPOINT_PATH", type=str, default="/home/zhaishichao/Data/Grounded_Segment_Anything/groundingdino_swint_ogc.pth")
    parser.add_argument("--rcnn_weight_path", type=str, default="/home/zhaishichao/Data/VLN/dependencies/mask_rcnn/maskrcnn_gibson.pth")
    parser.add_argument("--rcnn_yaml_path", type=str, default="/home/zhaishichao/Data/VLN/dependencies/mask_rcnn/mask_rcnn_R_50_FPN_3x.yaml")


    parser.add_argument("--BOX_THRESHOLD", type=float, default=0.6)
    parser.add_argument("--CONFIDENCE_TRESHOLE", type=float, default=0.6)
    parser.add_argument("--mask_rcnn_thre", type=float, default=0.6)

    parser.add_argument("--factor_0", type=float, default=1.5625)
    parser.add_argument("--factor_1", type=float, default=0.005859375)

    parser.add_argument("--depth_min_thre", type=float, default=0.1)
    parser.add_argument("--is_depth_estimation_laser", type=bool, default=True)

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()

class PreDepth(object):
    def __init__(self):
        self.width = args.depth_width
        self.height = args.depth_height
        self.dim = 5
        self.data = np.zeros((self.height, self.width, self.dim))

    def get_data(self):
        height = self.height
        width = self.width
        for i in range(height):
            for j in range(width):
                self.data[i, j, 3] = (j-width/2) / width * 2 * np.pi
                if i <= height / 4:
                    self.data[i, j, 0] = 6
                    self.data[i, j, 1] = np.absolute(j-width/2) / width * 2 * np.pi
                    self.data[i, j, 2] = np.absolute(i-height/2) / height * np.pi # 纬度
                elif i >= height / 2 - args.angle_depth / 45 * height / 4 and i <= height / 2 + args.angle_depth / 45 * height / 4:
                    self.data[i, j, 1] = np.absolute(j-width/2) / width * 2 * np.pi
                    self.data[i, j, 2] = np.absolute(i-height/2) / height * np.pi
                    # d_gt = d_cube / cos(jingdu) / cos(weidu+-...)
                    if j >= width / 2 - width / 8 and j <= width / 2 + width / 8:
                        self.data[i, j, 0] = 1
                    elif j >= width / 4 - width / 8 and j <= width / 4 + width / 8:
                        self.data[i, j, 0] = 2
                        self.data[i, j, 1] = np.absolute(j-width/4) / (width / 4) * 0.5 * np.pi
                    elif j >= 3 * width / 4 - width / 8 and j <= 3 * width / 4 + width / 8:
                        self.data[i, j, 0] = 3   
                        self.data[i, j, 1] = np.absolute(j-0.75*width) / (width / 4) * 0.5 * np.pi
                    else:
                        self.data[i, j, 0] = 4
                        if j < 0.25 * width:
                            temp = np.absolute(j-0)
                        else:
                            temp = np.absolute(j-width)
                        self.data[i, j, 1] = temp / (width / 4) * 0.5 * np.pi
                elif i >= 0.75 * height:
                    self.data[i, j, 0] = 5
                    self.data[i, j, 1] = np.absolute(j-width/2) / width * 2 * np.pi
                    self.data[i, j, 2] = np.absolute(i-height/2) / height * np.pi
                else:
                    self.data[i, j, 2] = np.absolute(i-height/2) / height * np.pi

                    if j >= width / 2 - width / 8 and j <= width / 2 + width / 8:
                        self.data[i, j, 1] = np.absolute(j-width/2) / width * 2 * np.pi
                        if np.arctan(1 / (1 / np.cos(self.data[i, j, 1]))) < self.data[i, j, 2]:
                            if i < 0.5 * height:
                                self.data[i, j, 0] = 6
                            else:
                                self.data[i, j, 0] = 5
                        else:
                            self.data[i, j, 0] = 1
                    elif j >= width / 4 - width / 8 and j <= width / 4 + width / 8:
                        self.data[i, j, 1] = np.absolute(j-width/4) / (width / 4) * 0.5 * np.pi
                        if np.arctan(1 / (1 / np.cos(self.data[i, j, 1]))) < self.data[i, j, 2]:
                            if i < 0.5 * height:
                                self.data[i, j, 0] = 6
                            else:
                                self.data[i, j, 0] = 5
                        else:
                            self.data[i, j, 0] = 2
                    elif j >= 3 * width / 4 - width / 8 and j <= 3 * width / 4 + width / 8:
                        self.data[i, j, 1] = np.absolute(j-0.75*width) / (width / 4) * 0.5 * np.pi
                        if np.arctan(1 / (1 / np.cos(self.data[i, j, 1]))) < self.data[i, j, 2]:
                            if i < 0.5 * height:
                                self.data[i, j, 0] = 6
                            else:
                                self.data[i, j, 0] = 5
                        else:
                            self.data[i, j, 0] = 3
                    else:
                        if j < 0.25 * width:
                            temp = np.absolute(j-0)
                        else:
                            temp = np.absolute(j-width)
                        self.data[i, j, 1] = temp / (width / 4) * 0.5 * np.pi
                        if np.arctan(1 / (1 / np.cos(self.data[i, j, 1]))) < self.data[i, j, 2]:
                            if i < 0.5 * height:
                                self.data[i, j, 0] = 6
                            else:
                                self.data[i, j, 0] = 5
                        else:
                            self.data[i, j, 0] = 4
                if self.data[i, j, 0] < 4.5:
                    self.data[i, j, 4] = np.cos(self.data[i, j, 2]) * np.cos(self.data[i, j, 1])
                else:
                    self.data[i, j, 4] = np.sin(self.data[i, j, 2])


pre_depth = PreDepth()
pre_depth.get_data()


ta_ls = []
for i in range(args.depth_width):
    ta = 1.5 * np.pi - i / args.depth_width * 2 * np.pi 
    if ta >= np.pi:
        ta = ta - 2*np.pi
    ta_ls.append(ta)
ta_ls_array = np.array(ta_ls)

coco_categories_mapping = {
        "chair": 56,
        "sofa": 57,  
        "plant": 58,
        "bed": 59, # mask：54
        "toilet": 61,
        "tv_monitor": 62
    }

