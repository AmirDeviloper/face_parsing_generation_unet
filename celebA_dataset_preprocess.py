import cv2
import numpy as np
import os.path as osp

from PIL import Image
from os import path, remove
from image_preprocess import dilation_on_mask
from global_parameters import MASK_PATH, IMAGE_PATH, FACE_SEP_MASK, IMAGE_HEIGHT, IMAGE_WIDTH

CELEBA_MASK_DIR_COUNT = 15
CELEBA_MASK_FILES_IN_DIR_COUNT = 2000

BAD_FACE_THRESHOLD = 170000
BAD_FACES_FILENAME = 'bad_faces.txt'

def _file_writer(txt):
    file1 = open(BAD_FACES_FILENAME, 'a')
    file1.write(f'{txt}.jpg\n')
    file1.close()

def _combine_masks():
    counter, total = 0, 0

    atts = ['skin', 1, 'hair', 2, 'nose', 3, 'l_lip', 4,
           'mouth', 4, 'u_lip', 4, 'r_ear', 5, 'l_ear', 5,
           'l_brow', 6, 'r_brow', 6, 'r_eye', 7, 'l_eye', 7]

    parts_need_dilation = ['l_lip', 'u_lip', 'mouth', 'r_brow', 'l_brow', 'l_eye', 'r_eye', 'nose']
    atts = { atts[i]: atts[i + 1] for i in range(0, len(atts), 2) }
    dilation_value = 2

    for i in range(CELEBA_MASK_DIR_COUNT):
        for j in range(i * CELEBA_MASK_FILES_IN_DIR_COUNT, (i + 1) * CELEBA_MASK_FILES_IN_DIR_COUNT):

            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

            for att, pixel_val in atts.items():
                total += 1
                savable = True
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(FACE_SEP_MASK, str(i), file_name)

                readed_mask_path = f'{FACE_SEP_MASK}\\{i}\\{j:05d}'
                if path.exists(f'{readed_mask_path}_hat.png') or path.exists(f'{readed_mask_path}_eye_g.png'):
                    savable = False
                    print(f'{j}.png has hat')
                    _file_writer(f'{j}_hat or glass')
                    break

                if path.exists(path):
                    mask_part = Image.open(path).convert('P')
                    counter += 1
                    sep_mask = np.array(mask_part)
                    if att in parts_need_dilation:
                        sep_mask = dilation_on_mask(sep_mask, att, dilation_value)

                    mask[sep_mask == 225] = pixel_val
                
            if np.count_nonzero(mask == 0) > BAD_FACE_THRESHOLD:
                savable = False
                print(f'{j}.png is a bad image.')
                _file_writer(f'{j}_bad image')

            if savable:
                cv2.imwrite(F'{MASK_PATH}/{j}.png', mask)

            print(j)

    print(counter, total)

def _remove_bad_images_masks():
    bad_faces_file = open(BAD_FACES_FILENAME, 'r')
    bad_faces = bad_faces_file.readlines()
    print(f'{len(bad_faces)} bad faces found in dataset.')

    for file_name in bad_faces:
        try:
            file_name = file_name.strip().split('_')[0].split('.')[0]
            print(file_name)
            img_path = f'{IMAGE_PATH}/{file_name}.jpg'

            if path.isfile(img_path):
                remove(f'{img_path}.jpg')

        except :
            print(f'error occoured on {file_name}')


_combine_masks()
_remove_bad_images_masks()


