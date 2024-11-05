import cv2
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models.vit import VisionTransformerEncoder
from lavis.common.registry import registry
from omegaconf import OmegaConf

import time


# total_time: 0.48947811126708984
def request_llm(rgb_image_ls, object_text):
    rgb_pil_ls = [Image.fromarray(temp_rgb[:, :, ::-1]) for temp_rgb in rgb_image_ls]    
    request_image_ls = [vis_processors["eval"](temp_rgb_pil).unsqueeze(0).to(device) for temp_rgb_pil in rgb_pil_ls]

    # [房间层面, object层面]
    question_ls = ["Is the current scene located in the room associated with {}? Please answer yes or no or not sure.".format(object_text), 
    "Is there the {} in the current scene? Please answer yes or no or not sure.".format(object_text)] 

    answer_ls = []
    
    request_image_combine = torch.cat([request_image_ls[0], request_image_ls[1], request_image_ls[2], request_image_ls[3],
    request_image_ls[0], request_image_ls[1], request_image_ls[2], request_image_ls[3]], dim=0)

    text_question_ls = [question_ls[0], question_ls[0], question_ls[0], question_ls[0],
    question_ls[1], question_ls[1], question_ls[1], question_ls[1]]

    text_answer_ls = model.predict_answers(samples={"image": request_image_combine, "text_input": text_question_ls}, inference_method="generate")
    num_answer_ls = [-1 if temp_answer not in answer_num_map else answer_num_map[temp_answer] for temp_answer in text_answer_ls]
    answer_ls = [max(num_answer_ls[0:4]), max(num_answer_ls[4:8])]
    return answer_ls
    
# Model Init
device = torch.device("cuda:0")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
answer_num_map = {"yes": 1, "no": 0}




