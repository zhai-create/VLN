import cv2
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

import time

# total_time: 0.4--0.6
def request_llm(rgb_image_ls, object_text):
    large_rgb = np.hstack((rgb_image_ls[2][:, int(rgb_image_ls[2].shape[1]//2):], rgb_image_ls[1], rgb_image_ls[0], rgb_image_ls[3], rgb_image_ls[2][:, :int(rgb_image_ls[2].shape[1]//2)]))
    large_rgb = large_rgb[:, :, ::-1]
    large_rgb = Image.fromarray(large_rgb)
    # large_rgb.save("large_rgb.jpg")
    # [房间层面, 相关容器层面, 引导landmark层面, object层面]
    question_ls = ["Is the current scene located in the room associated with {}? Please answer yes or no or not sure.".format(object_text), "Is there a receptacle related to {} in the current scene? Please answer yes or no or not sure.".format(object_text), "Is there a landmark in the current scene that leads to the {}? Please answer yes or no or not sure.".format(object_text), "Is there the {} in the current scene? Please answer yes or no or not sure.".format(object_text)] 
    request_image = vis_processors["eval"](large_rgb).unsqueeze(0).to(device) # 0.1s
    answer_ls = []
    
    # request the llm 
    # 1. 正向问题? 是否需要负向问题?
    # 2. 引入not sure?
    # 3. 是否换用其他模型?(时间，性能)
    # 4. 多专家的问答？
    for temp_question in question_ls:
        # q1: 0.1728653907775879, q2: 0.17457294464111328, 0.3042416572570801
        temp_answer = model.predict_answers(samples={"image": request_image, "text_input": temp_question}, inference_method="generate")
        if(temp_answer[0]=="yes"):
            answer_ls.append(1)
        elif(temp_answer[0]=="no"):
            answer_ls.append(0)
        else:
            print("temp_answer:", temp_answer[0])
            answer_ls.append(-1) # (yes:1, no:0, not sure: -1)
    return answer_ls


# Model Init
device = torch.device("cuda:1")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
