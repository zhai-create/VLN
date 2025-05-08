import cv2
import os
import base64
from PIL import Image
from io import BytesIO
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import torch
torch.manual_seed(1234)
import datetime


def get_qwen_response(save_name, prompt, history=None):
    # original_image = rgb_image_ls[0]
    # root_dir = "qwen_images/"
    # save_name = root_dir+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+".jpg"
    # cv2.imwrite(save_name, original_image)
    # image_source = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image_source)

    # buffered = BytesIO()
    # image.save(buffered, format='PNG')
    # image_bytes = base64.b64encode(buffered.getvalue())
    # image_str = str(image_bytes, 'utf-8')


    # Qwen-VL 直接支持 PIL 图像输入
    query = tokenizer.from_list_format([
        # {'image': image_str},  # 图像输入
        {"image": save_name},
        {'text': prompt},  # 文本输入
    ])
    response, history = model.chat(tokenizer, query=query, history=history)
    return response.strip("."), history


model_path = "Qwen/Qwen-VL-Chat"  # 或本地路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.float16, trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)


'''
prompt_1 = "You are a wheeled mobile robot working in an indoor environment.\
You are required to find an object goal:{} and navigate to it in as few steps as possible.\
I give you an observation image at the current moment.\
You need to estimate the probability of the object goal:{} appearing in the surrounding environment of the current image.\
I will guide you to answer this question through multiple rounds of questioning.\
First, please describe the objects present in this image, you have to answer with only a object list and do not answer any other text. Answer Example: ['bed', 'Mirror']"
prompt_2 = "Second, please state the room type described in the current picture.".format("bed", "bed")
prompt_3 = "Finall, please estimate the probability of the object goal:{} appearing in the surrounding environment of the current image.\
You can answer this question based on the above discussion.\
You have to answer with a value from 0 to 1 anyway.\
Answer only the value of probability and do not answer any other text.".format("bed")



res_1, history_1 = get_qwen_response(save_name="okok1.jpg", prompt=prompt_1, history=None)
print("res_1:", res_1)
print("history_1:", history_1)

res_2, history_2 = get_qwen_response(save_name="okok1.jpg", prompt=prompt_2, history=history_1)
print("res_2:", res_2)
print("history_2:", history_2)

res_3, history_3 = get_qwen_response(save_name="okok1.jpg", prompt=prompt_3, history=history_2)
print("res_3:", res_3)
print("history_3:", history_3)
'''