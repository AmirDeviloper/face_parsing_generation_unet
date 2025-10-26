from cmath import sqrt
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from global_parameters import *
from sklearn.cluster import KMeans

CLASSES = [ 'background', 'skin', 'hair', 'nose', 'lips', 'ears', 'brows', 'eyes' ]

def save_parts_and_masks(img, mask):
    final_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    partname_facepart_mask_list = []

    for i in range(NUM_CLASSES):
        rgb_code, part_name = RGB_CODES[i], CLASSES[i]

        temp_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        
        temp_mask[(mask == rgb_code).all(axis=2)] = 255

        filtered = temp_mask

        if part_name != 'background' and temp_mask.any():
            filtered = filtered.astype(np.uint8)

            _, bw = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY)

            face_part = cv2.bitwise_and(img, img, mask=bw)
            partname_facepart_mask_list.append((part_name, face_part, bw))


        final_mask[filtered[:, :] == 255] = rgb_code
        
    return final_mask, partname_facepart_mask_list

def save_final_parts(img, mask):
    partname_facepart_list = []
    bad_parts_name = CLASSES.copy()
    bad_parts_name.remove('background')
    good_parts_name = []

    for i in range(NUM_CLASSES):
        rgb_code, part_name = RGB_CODES[i], CLASSES[i]

        temp_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        temp_mask[(mask == rgb_code).all(axis=2)] = 255

        filtered = temp_mask

        target_area, boundry_area = None, None
        if part_name != 'background' and temp_mask.any():
            filtered = filtered.astype(np.uint8)

            _, bw = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY)
            face_part = cv2.bitwise_and(img, img, mask=bw)

            if part_name in ['ears']:
                target_area, boundry_area = good_ears_area(bw)

            elif part_name in ['skin', 'lips', 'nose', 'hair']:
                target_area, boundry_area = good_skin_lip_nose_hair_area(face_part, bw, part_name)

            elif part_name in ['eyes', 'brows']:
                target_area, boundry_area = good_eye_brow_area(face_part, bw, part_name)
                
        if target_area is not None:
            bad_parts_name.remove(part_name)
            good_parts_name.append(part_name)
            final_img = merge_with_skin(face_part, bw, part_name)
            label = get_label_of_face_part(part_name, target_area, boundry_area)
            partname_facepart_list.append((part_name, final_img, label))
            

    info = f'GOOD Parts For Segmention: {good_parts_name}\nBAD Parts For Segmention: {bad_parts_name}'
    return partname_facepart_list, info


def calculate_kmean(x1, x2, k=3):
    data = np.array(list(zip(x1, x2)))
    kmeans = KMeans(n_clusters=k).fit(data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    for i in range(k):
        print(f'{labels[i]} - ({centers[i, 0]}, {centers[i, 1]})')

    plt.scatter(data[:,0], data[:,1], c=labels)
    plt.scatter(centers[:,0], centers[:,1], marker='*', s=200, c='#050505')
    plt.show()


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def smallest_distance_label(points, new_point):
    closest_distance = float('inf')
    selected_label = ''
    for label in points:
        distance = calculate_distance(points[label], new_point)
        if distance < closest_distance:
            closest_distance = distance
            selected_label = label

    return selected_label


def get_label_of_face_part(_type, target_area, boundry_area):

    # part_areas calculated using kmeans on target & boundry areas.

    part_areas = {
        'ears': {
            'low':      (3667.799107142856, 5409.8928571428550),  
            'normal':   (4474.090163934426, 6924.1639344262285), 
            'high':     (5564.297872340425, 9023.0212765957440)
            },
                  
        'nose': {
            'low':      (7127.492753623189, 10745.231884057972),
            'normal':   (7866.634615384615, 12467.519230769230),
            'high':     (8578.448717948719, 14201.641025641027)
            },

        'lips': {
            'low':      (4004.303571428571, 5976.3571428571430),
            'normal':   (5103.788888888889, 8302.7111111111100),
            'high':     (7489.833333333333, 11809.500000000000)
            },

        'brows': {
            'low':      (1593.4306282722516, 2816.8691099476437),
            'normal':   (1961.6179138321995, 3654.9251700680275),
            'high':     (2586.8125000000005, 4800.6315789473670)
            },

        'eyes': {
            'low':      (1062.7669753086418, 1526.5462962962965),
            'normal':   (1255.9111747851002, 1818.9398280802290),
            'high':     (1474.5979899497486, 2129.2613065326630)
            },

        'skin': {
            'low':      (81119.56746031748, 106462.920634920620),
            'normal':   (91454.93965517242, 119162.637931034500),
            'high':     (105480.44565217392, 134800.19565217392)
            },

        'hair': {
            'low':      (44531.75308641973, 104938.033950617400),
            'normal':   (83542.43388429751, 204277.780991735550),
            'high':     (150079.93589743586, 251574.68065268075)
            }
        }

    target_areas = (target_area, boundry_area)
    label = smallest_distance_label(part_areas[_type], target_areas)

    return label
        

def show_img(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filled_blury_skin(ori_img, mask_img):
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    max_contour, _ = max_contour_and_area(mask_img)

    height, width = gray.shape[:2]
    x, y, w, h = cv2.boundingRect(max_contour)

    center_x_obj, center_y_obj = x + w // 2, y + h // 2
    center_x, center_y = width // 2, height // 2
    shift_x, shift_y = int(center_x - center_x_obj), int(center_y - center_y_obj)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    gray_centered = cv2.warpAffine(gray, M, (width, height))
    
    gray_centered, _ = fill_holes(gray_centered)

    return cv2.medianBlur(gray_centered, 3)


def mirror_ears(ori_img, mask_img):
    max_contour, _ = max_contour_and_area(mask_img)

    _, _, w, h = cv2.boundingRect(max_contour)

    one_ear_mask = np.zeros(mask_img.shape, np.uint8)
    cv2.drawContours(one_ear_mask, [max_contour], 0, (255, 255, 255), -1)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(ori_img, one_ear_mask)

    mirrored_ori = cv2.flip(result, 1)
    mirrored_mask = cv2.flip(one_ear_mask, 1)

    mirror_result = cv2.bitwise_and(mirrored_ori, mirrored_mask)
    or_masks = cv2.bitwise_or(mirror_result, result)

    return or_masks

def good_ears_area(mask_img):
    max_contour, target_area = max_contour_and_area(mask_img)

    ear_tresh = 3200
    if target_area > ear_tresh:
        _, _, w, h = cv2.boundingRect(max_contour)
        return (target_area, w * h)

    return (None, None)

def max_contour_and_area(img_mask):
    if len(img_mask.shape) == 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

    target_area = cv2.contourArea(max_contour)

    return max_contour, target_area

def fill_holes(gray):
    gray = np.array(gray)
    r_mean = int(np.mean(gray[:,:][gray[:, :] != 0]))
    mean_color = (r_mean, r_mean, r_mean)

    n = 2

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        moments = cv2.moments(contour)
        try:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

        except:
            pass

        center_point = (cx, cy)
        cv2.floodFill(gray, None, center_point, mean_color, loDiff=(n, n, n), upDiff=(n, n, n))

    return gray, mean_color[0]

def good_skin_lip_nose_hair_area(ori_img, mask_img, _type):
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    max_contour, target_area = max_contour_and_area(mask_img)

    height, width = gray.shape[:2]
    x, y, w, h = cv2.boundingRect(max_contour)

    center_x_obj, center_y_obj = x + w // 2, y + h // 2
    center_x, center_y = width // 2, height // 2
    shift_x, shift_y = int(center_x - center_x_obj), int(center_y - center_y_obj)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    gray_centered = cv2.warpAffine(gray, M, (width, height))

    if _type == 'skin':
        usable_face = False
        gray_centered, r_mean = fill_holes(gray)
        if gray_centered is not None:
            num_127 = (gray_centered == r_mean).sum()
            num_0 = (gray_centered == 0).sum()
            if num_127 < 23000 and num_0 < 200000:
                gray_centered = cv2.medianBlur(gray_centered, 3)
                usable_face = True

    right_img = gray_centered[:, width//2: ]
    left_img = gray_centered[:, : width//2]

    mirrored_right_half = cv2.flip(right_img, 1)

    _, left_img = cv2.threshold(left_img, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, mirrored_left_half = cv2.threshold(mirrored_right_half, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    xor_image = cv2.bitwise_xor(left_img, mirrored_left_half)
    diff_xor_count = cv2.countNonZero(xor_image)

    tresh_diff_values = { 'skin': (7500, 75000), 'lips': (220, 2600), 'nose': (425, 5500), 'hair': (999999999, 5000)}

    tresh_diff, tresh_area = tresh_diff_values[_type]

    if (diff_xor_count < tresh_diff and target_area > tresh_area):
        if _type != 'skin' or (usable_face and _type == 'skin'):

            return (target_area, w * h)

    return (None, None)

def is_middle_pupil(img):

    height, width = img.shape[: 2]
    third_width = width//3

    middle_third = img[:, third_width: 2 * third_width]

    height, width = middle_third.shape[: 2]
    split_parts = 5
    third_height = (height//split_parts) + 1

    middle_third = middle_third[third_height: split_parts * third_height, :]

    middle_third = middle_third[:, :] + abs((250 - middle_third.max()))
    _, binary = cv2.threshold(middle_third, 210, 255, cv2.THRESH_BINARY)

    black_pixels_middle = cv2.countNonZero(binary[third_height: split_parts * third_height, :])

    compare_tresh = 55

    return black_pixels_middle < compare_tresh

def good_eye_brow_area(img_ori, img_mask, _type):
    svable = False
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    tresh_diff_values = { 'eyes': 900, 'brows': 1500 }
    tresh_area = tresh_diff_values[_type]
    target_area = cv2.contourArea(max_contours[0])

    if target_area > tresh_area and len(max_contours) >= 2:
        x, y, w, h = cv2.boundingRect(max_contours[1])
            
        gray[y: y + h, x :x + w] = 0
        img_mask[y: y + h, x: x + w] = 0

        x, y, w, h = cv2.boundingRect(max_contours[0])
        new_img = gray[y: y + h, x: x + w]
        svable = (_type == 'eyes' and is_middle_pupil(new_img)) or _type == 'brows'

        if svable:
            return (target_area, (w * h))

    return (None, None)

def merge_with_skin(img, mask, _type = 'eyes', bg_skin=cv2.imread('__new_ex1.png')):

    first_img = img.copy()
    first_mask = mask.copy()

    _, img_w = img.shape[: 2]

    gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_contour)

    mask = mask[y: y + h, x: x + w]
    img = img[y: y + h, x: x + w]

    _, img_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    src_mask = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2RGB)

    if x > img_w // 2:
        src_mask = cv2.flip(src_mask, 1)
        img = cv2.flip(img, 1)

    if _type in ['eyes', 'brows', 'lips', 'nose']:
        weights = {'eyes': 130, 'brows': 150, 'lips': 200, 'nose': 200}
        weight = weights[_type]

        center = (weight//2, weight//2)
        bg_skin = bg_skin[: weight, : weight]

        output = cv2.seamlessClone(img, bg_skin, src_mask, center, cv2.MIXED_CLONE)
        final = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    if _type in ['eyes', 'brows']:
        w, h = output.shape[: 2]
        final = np.zeros((w, h * 2), dtype=np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        final[:, :h] = output
        final[:, h:] = cv2.flip(output, 1)

        return final

    elif _type in ['ears']:
        return mirror_ears(first_img, first_mask)

    elif _type in ['skin']:
        return filled_blury_skin(img, mask)

    elif _type in ['hair']:
        return cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

    return final


def classification_skins():
    '''
    hair
    0 - (42.82705643285948, 868.6401157376656)
    2 - (89.87017728132662, 2063.5752478728823)
    1 - (107.20777988815648, 3790.0865832083073)

    skin
    0 - (135.83031096467667, 1951.2581182079846)
    1 - (149.74482332569778, 2957.391946294144)
    2 - (156.11490109210433, 4615.355556764964)

    nose
    0 - (138.78424616574793, 2060.7685310323295)
    1 - (155.05820033855025, 3267.139808917654)
    2 - (154.58609956645327, 5246.694379224067)


    '''
    image_paths = sorted(glob(os.path.join('model_Attention_UNet_Conv2D_10000imgs_3epochs_transfer_learning', 'nose', "*.jpg")))

    x_list = []
    y_list = []
    for i in range(0, len(image_paths) - 1, 2):
        img = cv2.imread(image_paths[i], 0)
        nonzero_pixels = np.nonzero(img)
        mean = np.mean(img[nonzero_pixels])
        variance = np.var(img[nonzero_pixels])
        x_list.append(mean)
        y_list.append(variance)

    calculate_kmean(x_list, y_list)


