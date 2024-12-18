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
    time_1 = time.time()
    images = [Image.fromarray(temp_rgb[:, :, ::-1]) for temp_rgb in rgb_image_ls]  
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16) # [4, 3, 336, 336]
    print("=====> images_tensor.size<=====", images_tensor.size())
    time_2 = time.time()
    print("time2-time1:", time_2-time_1)

    # ====================> reasoning_stage_1 <====================
    output_id_room_1_ls = []
    output_id_object_1_ls = []
    with torch.inference_mode():
        for index in range(images_tensor.size(0)):
            output_id_room_1 = model.generate(
                input_id_room_1,
                images=images_tensor[index:index+1],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

            output_id_object_1 = model.generate(
                input_id_object_1,
                images=images_tensor[index:index+1],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
            output_id_room_1_ls.append(output_id_room_1)
            output_id_object_1_ls.append(output_id_object_1)

    output_room_1_ls = []
    output_object_1_ls = []
    for index in range(len(output_id_room_1_ls)):
        output_room_1 = tokenizer.batch_decode(output_id_room_1_ls[index], skip_special_tokens=True)[0].strip()
        output_object_1 = tokenizer.batch_decode(output_id_object_1_ls[index], skip_special_tokens=True)[0].strip()
        output_room_1_ls.append(output_room_1)
        output_object_1_ls.append(output_object_1)

    time_3 = time.time()
    print("time3-time2:", time_3-time_2)

    # ====================> prompt_process_stage_2 <====================
    input_id_room_2_ls = []
    for index in range(len(output_room_1_ls)):
        # --------------------> room_2 <--------------------
        qs_room_2 = "Based on the room type description: '{}', please infer the probability that the {} is in the current room and output it as a decimal in the range [0,1].".format(output_room_1_ls[index], object_text)
        qs_room_2 = DEFAULT_IMAGE_TOKEN + "\n" + qs_room_2

        conv_room_2 = conv_templates[conv_mode].copy()
        conv_room_2.append_message(conv_room_2.roles[0], qs_room_2)
        conv_room_2.append_message(conv_room_2.roles[1], None)
        prompt_room_2 = conv_room_2.get_prompt()

        input_id_room_2 = (
            tokenizer_image_token(prompt_room_2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        input_id_room_2_ls.append(input_id_room_2)

    input_id_object_2_ls = []
    for index in range(len(output_object_1_ls)):
        # --------------------> object_2 <--------------------
        qs_object_2 = "Based on the detailed description of the objects: '{}', please infer the probability of the {} being around these objects and output it as a decimal in the range [0,1].".format(output_object_1_ls[index], object_text)
        qs_object_2 = DEFAULT_IMAGE_TOKEN + "\n" + qs_object_2

        conv_object_2 = conv_templates[conv_mode].copy()
        conv_object_2.append_message(conv_object_2.roles[0], qs_object_2)
        conv_object_2.append_message(conv_object_2.roles[1], None)
        prompt_object_2 = conv_object_2.get_prompt()

        input_id_object_2 = (
            tokenizer_image_token(prompt_object_2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        input_id_object_2_ls.append(input_id_object_2)
    
    time_4 = time.time()
    print("time4-time3:", time_4-time_3)
    
    # ====================> reasoning_stage_2 <====================
    output_id_room_2_ls = []
    output_id_object_2_ls = []
    with torch.inference_mode():
        for index in range(images_tensor.size(0)):
            output_id_room_2 = model.generate(
                input_id_room_2_ls[index],
                images=images_tensor[index:index+1],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

            output_id_object_2 = model.generate(
                input_id_object_2_ls[index],
                images=images_tensor[index:index+1],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
            output_id_room_2_ls.append(output_id_room_2)
            output_id_object_2_ls.append(output_id_object_2)


    output_room_2_ls = []
    output_object_2_ls = []
    for index in range(len(output_id_room_2_ls)):
        output_room_2 = tokenizer.batch_decode(output_id_room_2_ls[index], skip_special_tokens=True)[0].strip()
        output_object_2 = tokenizer.batch_decode(output_id_object_2_ls[index], skip_special_tokens=True)[0].strip()
        # =====> 数据类型转换 <=====
        try:
            output_room_2 = eval(output_room_2)
        except:
            output_room_2 = 0
        try:
            output_object_2 = eval(output_object_2)
        except:
            output_object_2 = 0
        # =====> 数据类型转换 <=====
        output_room_2_ls.append(output_room_2)
        output_object_2_ls.append(output_object_2)

    print("output_room_2_ls:", output_room_2_ls)
    print("output_object_2_ls:", output_object_2_ls)

    room_type_score = np.mean(output_room_2_ls)
    object_details_score = np.mean(output_object_2_ls)
    answer_ls = [room_type_score, object_details_score]
    time_5 = time.time()
    print("time_5-time_4:", time_5-time_4)
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
    device="cuda"
)
conv_mode = "llava_v1"
print("llava init success!")

# ====================> 2. prompt_process <====================
# --------------------> 2.1 room_1 <--------------------
qs_room_1 = "Describe the room type in the image."
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
qs_object_1 = "Describe the objects with details in the image."
qs_object_1 = DEFAULT_IMAGE_TOKEN + "\n" + qs_object_1

conv_object_1 = conv_templates[conv_mode].copy()
conv_object_1.append_message(conv_object_1.roles[0], qs_object_1)
conv_object_1.append_message(conv_object_1.roles[1], None)
prompt_object_1 = conv_object_1.get_prompt()
'''prompt_object_1: A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
Describe the objects with details in the image. ASSISTANT:'''

input_id_object_1 = (
    tokenizer_image_token(prompt_object_1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .cuda()
)

print("original prompts init success!")



# ====================> small_test <====================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
rgb_1 = cv2.imread("okok1.jpg")
rgb_2 = cv2.imread("okok2.jpg")
rgb_3 = cv2.imread("okok3.jpg")
rgb_4 = cv2.imread("okok4.jpg")

rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
start_time = time.time()
request_llm(rgb_image_ls, object_text="chair")
end_time = time.time()

print("=====> delta_time11111 <=====", end_time-start_time)


rgb_image_ls = [rgb_1, rgb_2, rgb_3, rgb_4]
start_time = time.time()
request_llm(rgb_image_ls, object_text="chair")
end_time = time.time()

print("=====> delta_time22222 <=====", end_time-start_time)

# bed:
# output_room_1_ls: [0.99, 0.99, 0.0, 0.5]
# output_object_1_ls: [0.75, 0.8, 0.0, 0.0]
# 第一次调用时间: 26.35559844970703(22.423075914382935+3.852424383163452)
# 第二次调用时间: 22.629162788391113(18.71650457382202+3.8333325386047363)

# chair: 
# output_room_1_ls: [0.0, 0.5, 0.5, 0.5]
# output_object_1_ls: [0.0, 0, 0.0, 0.1]
# 第一次调用时间: 22.660972118377686(19.26382827758789+3.3207974433898926)
# 第二次调用时间: 18.94590735435486(15.53856635093689+3.328800678253174)


