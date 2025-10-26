import tensorflow as tf
import numpy as np
import cv2

from global_parameters import *
from tensorflow.keras.preprocessing.image import img_to_array

def read_image_mask(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x/255.0
    x = x.astype(np.float32)

    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (IMAGE_WIDTH, IMAGE_HEIGHT))
    y = y.astype(np.int32)

    return x, y


def read_image(z, read_type=cv2.IMREAD_COLOR):
    """ Image """
    x = cv2.imread(z, read_type)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return x

def _read_image_to_array(x):
    x = read_image(x)
    x = img_to_array(x)
    
    return x / 255

def _image_to_tensor(func, x, y, for_autoencoder = False):
    y_type, y_depth = (tf.float32, IMAGE_CHANNELS) if for_autoencoder else (tf.int32, NUM_CLASSES)
    image, mask = tf.numpy_function(func, [x, y], [tf.float32, y_type])
    if not for_autoencoder:
        mask = tf.one_hot(mask, y_depth)

    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    mask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, y_depth])

    return image, mask

def _autoencoder_preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return _read_image_to_array(x), _read_image_to_array(y)

    image_in, image_out = _image_to_tensor(f, x, y, True)

    return image_in, image_out

def _unet_preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)
    image, mask = _image_to_tensor(f, x, y, False)

    return image, mask

def tf_dataset(X, Y, for_autoencoder = False, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    func = _autoencoder_preprocess if for_autoencoder else _unet_preprocess
    ds = ds.shuffle(buffer_size=5000).map(func) 
    ds = ds.batch(batch).prefetch(2)
    return ds


def dilation_on_mask(mask_img, mask_name, k):
    
    mask_names = ['l_lip', 'u_lip', 'mouth', 'nose', 'r_eye', 'l_eye', 'l_brow', 'r_brow']
    kernels = [LIPS_KERNEL, LIPS_KERNEL, LIPS_KERNEL, NOSE_KERNEL, R_EYE, L_EYE, L_BROW, R_BROW]
    kernel = None

    for i, mask in enumerate(mask_names):
        if mask == mask_name:
            kernel = kernels[i]
            break

    return cv2.dilate(mask_img, kernel, iterations=k)

def grayscale_to_rgb(mask, RGB_CODES):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for _, pixel in enumerate(mask.flatten()):
        output.append(RGB_CODES[pixel])

    output = np.reshape(output, (h, w, 3))
    return output