# Percption Module
## Get 2d-laser
* laser_utils.py
    * def laser_filter(laser_2d):
        * Filter the 2d-laser and get corresponding angle
        * param laser_2d: original laser
        * return laser_2d_filtered:  filtered 2d-laser based on gradient 
        * return laser_2d_filtered_angle: corresponding angle of the 2d-laser

    * def get_laser_point(depth):
        * Get the filtered 2d-laser
        * param depth: depth after fixed
        * return point_for_close_loop_detection: 2d-laser for ring
        * return laser_2d_filtered:  robot's current 2d-laser
        * return laser_2d_filtered_angle: corresponding angle of the 2d-laser

## Get the candidate frontier nodes
* frontier_utils.py
    * def predict_frontier(thre1, laser_2d_filtered, laser_2d_filtered_angle):
        * Get the candidate intention nodes, which need the pos transformer and the selection
        * param rgb_image_ls: [front_img, left_img, behind_img, right_img]
        * param depth: depth after fixed
        * param object_text: object-goal
        * return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}
    <!-- * Input:  Gradient threshold for detection, laser_2d_filtered, laser_2d_filtered_angle
    * Output: candidate frontier ndoes -->


## Get intention nodes
* intention_utils_dino.py
    * def object_detect(rgb_image_ls, depth, object_text):
        * Based on DINO and SAM, get the candidate intention nodes, which need the pos transformer and the selection
        * param rgb_image_ls: [front_img, left_img, behind_img, right_img]
        * param depth: depth after fixed
        * param object_text: object-goal
        * return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}

    * DINO Init
    * SAM Init

* intention_utils_rcnn.py
    * def object_detect(rgb_image_ls, depth, object_text):
        * Based on Mask-RCNN, get the candidate intention nodes, which need the pos transformer and the selection
        * param rgb_image_ls: [front_img, left_img, behind_img, right_img]
        * param depth: depth after fixed
        * param object_text: object-goal
        * return detect_res_pos_dict: {key: score, value: [rela_pos1, rela_pos2, ...]}

    * Mask-RCNN Init

## Get multiple Q&A tags based on LLM
* intention_utils_blip.py
    * def request_llm(rgb_image_ls, object_text):
        * Based on blip_vqa, obtain confidence scores at the room and object levels through Q&A
        * param rgb_image_ls: [front_img, left_img, behind_img, right_img]
        * param object_text: object-goal
        * return answer_ls: [room_type_score, object_type_score]

    * BLIP_vqa Init


* intention_utils_llava_easy.py
    * def request_llm(rgb_image_ls, object_text):
        * Based on LLaVa, obtain confidence scores at the room and object levels through a single round of Q&A 
        * param rgb_image_ls: [front_img, left_img, behind_img, right_img]
        * param object_text: object-goal
        * return answer_ls: [room_type_score, object_type_score]

    * LLaVa Init


## Revelent Utility Tools
* tools.py
    * def fix_depth(depth):
        * Fix the sensor depth
        * param depth: original sensor depth
        * return depth: depth with real dis
    
    * def resize_matrix(matrix, new_shape):
        * Reduce the size of the corresponding matrix and output the values of each position in the matrix according to the relative ratio of the positions of the values in the original matrix on the rows and columns
        * param matrix: the imput 2d matrix(element: 0 or 1)
        * param new_shape: the output matrix shape (new_row_num, new_col_num)
        * return: resized matrix

    * def depth_estimation(large_mask, depth):
        * Get the relative pos of the detected object
        * param large_mask: mask result corresponding to the panoramic depth
        * param depth: fixed depth
        * return res_depth_2d_cx, res_depth_2d_cy: relative pos of the object corresponding to the current robot

    * def sam_show_mask(mask, image0):
        * Utility: Show the mask on the rgb image

    * def transform_rgb_bgr(image):
        * Transform rgb to bgr
        * param image: rgb
        * return image: bgr

    * def get_rgb_image_ls(env):
        * Get the image list
        * param env: habitat_env
        * return image_ls: [rgb_1, rgb_2, rgb_3, rgb_4]