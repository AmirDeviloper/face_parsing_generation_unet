import os
import numpy as np

from glob import glob
from cv2 import imwrite
from global_parameters import RGB_CODES
from image_preprocess import grayscale_to_rgb
from image_postprocess import save_final_parts, save_parts_and_masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def save_results_img_mask_mask(image_x, mask, pred, save_image_path):
    image_dir = save_image_path[: save_image_path.rfind('/')]
    print(save_image_path)
    create_dir(image_dir)

    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, RGB_CODES)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, RGB_CODES)
    filtered_pred, partname_facepart_mask_list = save_parts_and_masks(image_x, pred)

    line = np.ones((image_x.shape[0], 10, 3)) * 255

    cat_images = np.concatenate([image_x, line, mask, line, filtered_pred], axis=1)
    imwrite(save_image_path, cat_images)

    for (partname, facepart, facepart_mask) in partname_facepart_mask_list:
        imwrite(f'{image_dir}\{partname}.jpg', facepart)
        imwrite(f'{image_dir}\{partname}_mask.jpg', facepart_mask)

def save_results_img_mask_mask_2(image_x, mask, pred, save_image_path):
    img_name = save_image_path[save_image_path.rfind('/') + 1: save_image_path.rfind('.')]
    image_dir = save_image_path[: save_image_path.rfind('/')]
    
    image_dir = image_dir[: image_dir.rfind('/')]

    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, RGB_CODES)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, RGB_CODES)
    filtered_pred, partname_facepart_mask_list = save_parts_and_masks(image_x, pred)

    line = np.ones((image_x.shape[0], 10, 3)) * 255
    
    create_dir(f'{image_dir}\comparison')
    cat_images = np.concatenate([image_x, line, mask, line, filtered_pred], axis=1)
    imwrite(f'{image_dir}\comparison\{img_name}.jpg', cat_images)

    for (partname, facepart, facepart_mask) in partname_facepart_mask_list:
        create_dir(f'{image_dir}\{partname}')
        imwrite(f'{image_dir}\{partname}\{img_name}_{partname}.jpg', facepart)
        imwrite(f'{image_dir}\{partname}\{img_name}_{partname}_mask.jpg', facepart_mask)

def save_final_results(image_x, pred, save_image_path):
    img_name = save_image_path[save_image_path.rfind('/') + 1: save_image_path.rfind('.')]
    image_dir = save_image_path[: save_image_path.rfind('/')]
    
    image_dir = image_dir[: image_dir.rfind('/')]

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, RGB_CODES)

    partname_facepart_mask_list, info = save_final_parts(image_x, pred)

    with open(f'{image_dir}\info.txt', 'a') as info_file:
        sep = '---' * 10
        info_file.write(f'{img_name}.jpg Segmention Info\n')
        info_file.write(f'{info}\n')
        info_file.write(f'{sep}\n')
    
    for (partname, final_img, label) in partname_facepart_mask_list:
        create_dir(f'{image_dir}\{partname}\{label}')
        imwrite(f'{image_dir}\{partname}\{label}\{img_name}_{partname}.jpg', final_img)

def save_results_img_img(image_x, image_y, save_image_path):

    line = np.ones((image_x.shape[0], 10, 3)) * 255

    cat_images = np.concatenate([image_x, line, image_y], axis=1)
    imwrite(save_image_path, cat_images)

