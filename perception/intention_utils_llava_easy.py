# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import cv2
import torch
import time
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
# from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image


# 1. 图片如何并行化处理?(用目前的方法只能推理第一张图片)
# 2. prompt如何并行化? (input_id_token需要同样的长度)
# 3. 较难问题减少推理时间?(第一轮问题，一张图片，两个prompt回答需要3.7901968955993652)


def request_llm(rgb_image_ls, object_text):
    # ====================> image relevant <====================
    images = [Image.fromarray(temp_rgb[:, :, ::-1]) for temp_rgb in rgb_image_ls]  
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16) # [4, 3, 336, 336]

    # ====================> 2. prompt_process <====================
    # --------------------> 2.1 room_1 <--------------------
    qs_room_1 = "Infer the probability that the {} is in the current room and output it as a decimal in the range [0,1].".format(object_text)
    qs_room_1 = DEFAULT_IMAGE_TOKEN + "\n" + qs_room_1

    conv_room_1 = conv_templates[conv_mode].copy()
    conv_room_1.append_message(conv_room_1.roles[0], qs_room_1)
    conv_room_1.append_message(conv_room_1.roles[1], None)
    prompt_room_1 = conv_room_1.get_prompt()

    input_id_room_1 = (
        tokenizer_image_token(prompt_room_1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # --------------------> 2.2 object_1 <--------------------
    qs_object_1 = "Infer the probability of the {} around these objects in image and output it as a decimal in the range [0,1].".format(object_text)
    qs_object_1 = DEFAULT_IMAGE_TOKEN + "\n" + qs_object_1

    conv_object_1 = conv_templates[conv_mode].copy()
    conv_object_1.append_message(conv_object_1.roles[0], qs_object_1)
    conv_object_1.append_message(conv_object_1.roles[1], None)
    prompt_object_1 = conv_object_1.get_prompt()

    input_id_object_1 = (
        tokenizer_image_token(prompt_object_1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    # ============================================================

    # tensor_concat
    input_ids = torch.cat((input_id_room_1, input_id_room_1, input_id_room_1, input_id_room_1, 
                           input_id_object_1, input_id_object_1, input_id_object_1, input_id_object_1), dim=0)

    large_images_tensor = torch.cat((images_tensor, images_tensor), dim=0)
    # large_images_tensor = large_images_tensor.to(torch.float32)
    # ====================> reasoning_stage_1 <====================
    with torch.inference_mode():    
        output_ids= model.generate(
            input_ids,
            images=large_images_tensor,
            image_sizes=image_sizes+image_sizes,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    outputs_all_str_ls = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("outputs_all_str_ls:", outputs_all_str_ls)

    output_room_str_ls = outputs_all_str_ls[0:4]
    output_room_score_ls = []
    for index in range(len(output_room_str_ls)):
        temp_str = (output_room_str_ls[index].strip()).strip(".")
        try:
            output_room_score = eval(temp_str)
        except:
            output_room_score = 0
            temp_str_ls = temp_str.split(" ")[::-1]
            for temp_index in range(len(temp_str_ls)):
                try:
                    output_room_score = eval(temp_str_ls[temp_index].strip(','))
                    break
                except:
                    continue
        output_room_score_ls.append(output_room_score)

    output_object_str_ls = outputs_all_str_ls[4:8]
    output_object_score_ls = []
    for index in range(len(output_object_str_ls)):
        temp_str = (output_object_str_ls[index].strip()).strip(".")
        try:
            output_object_score = eval(temp_str)
        except:
            output_object_score = 0
            temp_str_ls = temp_str.split(" ")[::-1]
            for temp_index in range(len(temp_str_ls)):
                try:
                    output_object_score = eval(temp_str_ls[temp_index].strip(','))
                    break
                except:
                    continue
        output_object_score_ls.append(output_object_score)

    print("output_room_score_ls:", output_room_score_ls)
    print("output_object_score_ls:", output_object_score_ls)

    try:
        room_type_score = np.mean(output_room_score_ls)
    except:
        room_type_score = 0

    try:
        object_details_score = np.mean(output_object_score_ls)
    except:
        object_details_score = 0
    answer_ls = [room_type_score, object_details_score]
    return answer_ls



# ====================> init_model <====================
# ====================> 1. model_type <====================
model_path = "/home/zhaishichao/Data/VLN/dependencies/LLaVA/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path) # llava-v1.5-7b
# disable_torch_init()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device="cuda",
    device_map="cuda"
)
# model = model.to(torch.float32)
conv_mode = "llava_v1"
print("llava init success!")


# # ====================> small_test <====================

# rgb_1 = cv2.imread("okok1.jpg")
# rgb_2 = cv2.imread("okok2.jpg")
# rgb_3 = cv2.imread("okok3.jpg")
# rgb_4 = cv2.imread("okok4.jpg")

# rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
# start_time = time.time()
# print(request_llm(rgb_image_ls, object_text="bed"))
# end_time = time.time()

# print("=====> delta_time11111 <=====", end_time-start_time)


# rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
# start_time = time.time()
# print(request_llm(rgb_image_ls, object_text="bed"))
# end_time = time.time()

# print("=====> delta_time22222 <=====", end_time-start_time)

# # ==============> init(无COT, 遍历图片，遍历prompt) <===============
# # bed:
# # output_room_1_ls: [0.9, 0.8, 0.0, 0.5]
# # output_object_1_ls: [0, 0, 0.0, 0.1]
# # 第一次调用时间: 9.164653301239014
# # 第二次调用时间: 5.34675669670105


# # chair: 
# # output_room_1_ls: [0.1, 0.5, 0.5, 0.5]
# # output_object_1_ls: [0.1, 0, 0.1, 0.1]
# # 第一次调用时间: 8.404532432556152
# # 第二次调用时间: 4.561633348464966

# # ==============> 无COT, 并行化 <===============
# # bed:
# # output_room_1_ls: [0.9, 0.8, 0.0, 0.5]
# # output_object_1_ls: [0.99, 0.99, 0.1, 0.1]
# # 第一次调用时间: 4.446522235870361
# # 第二次调用时间: 2.229058265686035


# # chair: 
# # output_room_1_ls: [0.1, 0.5, 0.5, 0.5]
# # output_object_1_ls: [0.1, 0.85, 0.15, 0.85]
# # 第一次调用时间: 4.552340745925903
# # 第二次调用时间: 2.300286054611206

# # 当前平均调用时间：2.26s



