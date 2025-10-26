import os
import numpy as np
import tensorflow as tf

from image_preprocess import *
from dataset_handler import create_dir, load_dataset, save_final_results

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Hyperparameters """
test_len = 300

""" Paths """
model_name = 'model_Attention_UNet_Conv2D_10000imgs_3epochs_transfer_learning1.h5'
model_path = os.path.join("files", model_name)

""" Directory for storing files """
dirc_name = f'{model_name}_results'
create_dir(dirc_name)

""" Loading the dataset """
(_, _), (_, _), (test_x, _) = load_dataset(DATASET_PATH)
(test_x, _) = (test_x [0: test_len], _)


""" Load the model """
model = tf.keras.models.load_model(model_path, compile=False)

i = 0
""" Prediction & Evaluation """
for x in test_x:
    print(i)
    i += 1
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Reading the image and mask"""
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image/255.0
    image = image.astype(np.float32)

    image_x = read_image(x)
    image = np.expand_dims(image, axis=0) ## [1, H, W, 3]

    """ Prediction """
    pred = model.predict(image, verbose=0)[0]
    pred = np.argmax(pred, axis=-1) ## [0.1, 0.2, 0.1, 0.6] -> 3
    pred = pred.astype(np.int32)

    """ Save the results """
    name = name.split('\\')[-1]
    save_path = f'{dirc_name}/{name}'
    save_image_path = f"{save_path}/{name}.jpg"
    
    save_final_results(image_x, pred, save_image_path)

