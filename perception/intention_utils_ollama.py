import cv2
import base64
from PIL import Image
from io import BytesIO
import ollama
import json

# def get_vlm_response(rgb_image_ls, prompt):
#     original_image = rgb_image_ls[0]
#     image_source = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image_source)
    
#     buffered = BytesIO()
#     image.save(buffered, format='PNG')
#     image_bytes = base64.b64encode(buffered.getvalue())
#     image_str = str(image_bytes, 'utf-8')

#     response = ollama.chat(
#         model='llama3.2-vision',
#         messages=[{
#             'role': 'user',
#             'content': prompt,
#             'images': [image_str]
#         }],
#         options={
#             'temperature': 0.0,  # 设置为 0 消除随机性
#             'top_p': 0.0,        # 设置为 0 进一步确保确定性
#             'seed': 12345        # 设置一个固定的随机种子，以保证可复现性
#         }
#     )
#     return response.message.content.strip(".")


# def get_vlm_response(rgb_image_ls, prompt):
#     original_image = rgb_image_ls[0]
#     _, buffer = cv2.imencode(".png", original_image)  # 直接编码为 PNG 字节流
#     image_str = base64.b64encode(buffer).decode("utf-8")  # 转为 base64
    
#     response = ollama.chat(
#         model='llama3.2-vision',
#         messages=[{
#             'role': 'user',
#             'content': prompt,
#             'images': [image_str]
#         }],
#         options={'temperature': 0, 'top_p': 0, 'seed': 12345}
#     )
#     return response.message.content.strip(".")

def get_vlm_response(image_str, prompt):
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_str]
        }],
        options={'temperature': 0, 'top_p': 0, 'seed': 12345}
    )
    return response.message.content.strip(".")

def clean_response(text):
    # 移除首尾空白 + 替换换行符 + 移除特殊字符
    return text.strip().replace("\n", "").replace("\r", "")

'''
test_img = cv2.imread("okok1.jpg")

original_image = test_img
_, buffer = cv2.imencode(".png", original_image)  # 直接编码为 PNG 字节流
image_str = base64.b64encode(buffer).decode("utf-8")  # 转为 base64

res_2 = get_vlm_response(image_str, prompt="What is the probability of {} appearing around this image? You should answer this question based on objects and room type in this image. You have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.".format("sofa"))
# res_2 = get_vlm_response(image_str, prompt="Based on the objects and room type in this image, what is the probability of {} appearing around this image? Answer only a number from 0 to 1 and do not answer any other text.".format("toilet"))

print(res_2)
print(type(res_2))
print(eval(res_2))
'''


# res_1 = get_vlm_response(image_str, prompt="Describe the objects present in this image, you have to answer with only a object list and do not answer any other text. Answer Example: ['bed', 'Mirror']}")
# res_1 = clean_response(res_1)
# res_1 = json.dumps(eval(res_1))
# res_2 = get_vlm_response(image_str, prompt="What is the probability of {} appearing around this this image? You should answer this question based on this image and its corresponding object list: {}. You have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.".format("tv", res_1))


# print(res_1)
# print(res_2)
# print(type(res_2))

# res_2 = get_vlm_response(rgb_image_ls=[test_img], prompt="What is the probability of {} appearing around this this image? You have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.".format("bed"))
# print(res_2)

# res_1 = get_vlm_response(rgb_image_ls=[test_img], prompt="Describe the objects present in this image.")
# print(res_1)