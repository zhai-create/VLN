import cv2
import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

import time

def request_llm(rgb_image_ls, object_text):
    large_rgb = np.hstack((rgb_image_ls[2][:, int(rgb_image_ls[2].shape[1]//2):], rgb_image_ls[1], rgb_image_ls[0], rgb_image_ls[3], rgb_image_ls[2][:, :int(rgb_image_ls[2].shape[1]//2)]))
    large_rgb = large_rgb[:, :, ::-1]
    print("=====> shape <=====", large_rgb.shape)
    large_rgb = Image.fromarray(large_rgb)
    large_rgb.save("large_rgb.jpg")
    # [房间层面, 相关容器层面, 引导landmark层面, object层面]
    question_ls = ["Is the current scene located in the room associated with {}? Please answer yes or no or not sure.".format(object_text), "Is there a receptacle related to {} in the current scene? Please answer yes or no.".format(object_text), "Is there a landmark in the current scene that leads to the {}? Please answer yes or no.".format(object_text), "Is there the {} in the current scene? Please answer yes or no or not sure.".format(object_text)] 
    start_time = time.time()
    request_image = vis_processors["eval"](large_rgb).unsqueeze(0).to(device) # 0.1s
    end_time = time.time()
    print("end-start:", end_time-start_time)
    answer_ls = []
    
    # request the llm 
    # 1. 正向问题? 是否需要负向问题?
    # 2. 引入not sure?
    # 3. 是否换用其他(时间，性能)
    # 4. 多专家的问答？
    for temp_question in question_ls:
        start_t = time.time()
        # q1: 0.1728653907775879, q2: 0.17457294464111328, 0.3042416572570801
        temp_answer = model.predict_answers(samples={"image": request_image, "text_input": temp_question}, inference_method="generate")
        end_t = time.time()
        print("delta_time:", end_t-start_t)
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

rgb_1 = cv2.imread("okok1.jpg")
rgb_2 = cv2.imread("okok2.jpg")
rgb_3 = cv2.imread("okok3.jpg")
rgb_4 = cv2.imread("okok4.jpg")

answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "chair") # [0, 1, -1](object, infrastructure)-->[0, 1, -1, 1](receptacle)
print(answer_ls)
answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "bed") # [1, 1, 0]-->[1, 1, 0, 1]
print(answer_ls)
answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "plant") # [0, 1, 0]-->[0, 0, 0, 0]
print(answer_ls)
answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "toilet") # [0, 1, 0]-->[0, 0, 0, 0]
print(answer_ls)
answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "tv_monitor") # [0, 0, 1]-->[0, 0, 1, 0]
print(answer_ls)
answer_ls = request_llm([rgb_1, rgb_2, rgb_3,rgb_4], "sofa") # [0, 1, 0]-->[0, 0, 0, 0]
print(answer_ls)