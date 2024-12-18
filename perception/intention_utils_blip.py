import cv2
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models.vit import VisionTransformerEncoder
from lavis.common.registry import registry
from omegaconf import OmegaConf

import time
from env_tools.arguments import args as env_args


# total_time: 0.48947811126708984
def request_llm(rgb_image_ls, object_text):
    rgb_pil_ls = [Image.fromarray(temp_rgb[:, :, ::-1]) for temp_rgb in rgb_image_ls]    
    request_image_ls = [vis_processors["eval"](temp_rgb_pil).unsqueeze(0).to(device) for temp_rgb_pil in rgb_pil_ls]

    # [房间层面, object层面]
    # question_ls = ["Is the current scene in a room that is semantically related to the {}? Please answer yes or no or not sure.".format(object_text), 
    # "Is there an object in the current scene that are semantically related to the {}? Please answer yes or no or not sure.".format(object_text)] 

    question_ls = ["Is the current scene located in the room associated with {}? Please answer yes or no or not sure.".format(object_text), 
    "Is there the {} in the current scene? Please answer yes or no or not sure.".format(object_text)] 


    answer_ls = []
    
    if(env_args.is_one_rgb==False):
        request_image_combine = torch.cat([request_image_ls[0], request_image_ls[1], request_image_ls[2], request_image_ls[3],
        request_image_ls[0], request_image_ls[1], request_image_ls[2], request_image_ls[3]], dim=0)

        text_question_ls = [question_ls[0], question_ls[0], question_ls[0], question_ls[0],
        question_ls[1], question_ls[1], question_ls[1], question_ls[1]]

        text_answer_ls = model.predict_answers(samples={"image": request_image_combine, "text_input": text_question_ls}, inference_method="generate")
        num_answer_ls = [-1 if temp_answer not in answer_num_map else answer_num_map[temp_answer] for temp_answer in text_answer_ls]
        answer_ls = [max(num_answer_ls[0:4]), max(num_answer_ls[4:8])]
    else:
        request_image_combine = torch.cat([request_image_ls[0],
        request_image_ls[0]], dim=0)

        text_question_ls = [question_ls[0],
        question_ls[1]]

        text_answer_ls = model.predict_answers(samples={"image": request_image_combine, "text_input": text_question_ls}, inference_method="generate")
        num_answer_ls = [-1 if temp_answer not in answer_num_map else answer_num_map[temp_answer] for temp_answer in text_answer_ls]
        answer_ls = [num_answer_ls[0], num_answer_ls[1]]
    return answer_ls
    

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1, 2, 3'


# Model Init
# device = torch.device("cuda:0")
device = torch.device("cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
answer_num_map = {"yes": 1, "no": 0}




# rgb_1 = cv2.imread("okok1.jpg")
# rgb_2 = cv2.imread("okok2.jpg")
# rgb_3 = cv2.imread("okok3.jpg")
# rgb_4 = cv2.imread("okok4.jpg")

# rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
# answer_ls = request_llm(rgb_image_ls, object_text="tv")
# print(answer_ls)




