import cv2
import base64
from PIL import Image
from io import BytesIO
import ollama

def get_vlm_response(rgb_image_ls, prompt):
    original_image = rgb_image_ls[0]
    image_source = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_source)
    
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    image_bytes = base64.b64encode(buffered.getvalue())
    image_str = str(image_bytes, 'utf-8')

    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_str]
        }]
    )
    return response.message.content.strip(".")


# test_img = cv2.imread("okok1.jpg")
# res_1 = get_vlm_response(rgb_image_ls=[test_img], prompt="Describe the objects present in this image, you have to answer with only a object list and do not answer any other text. Answer Example: ['bed', 'Mirror']}")
# res_2 = get_vlm_response(rgb_image_ls=[test_img], prompt="What is the probability of {} appearing around this this image? You should answer this question based on this image and its corresponding object list: {}. You have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.".format("tv", res_1))


# res_2 = get_vlm_response(rgb_image_ls=[test_img], prompt="What is the probability of {} appearing around this this image? You have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.".format("tv"))
# print(res_1)
# print(res_2)