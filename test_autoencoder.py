import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf

from image_preprocess import *
from global_parameters import *
from sklearn.metrics import mean_squared_error
from dataset_handler import create_dir, load_dataset, save_results_img_img


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Hyperparameters """
test_len = 100

""" Paths """
model_name = 'model_autoencoder_20001imgs_8epochs.h5'
model_path = os.path.join("files", model_name)

""" Directory for storing files """
dirct_name = f"{model_name}_results" 
create_dir(dirct_name)

""" Loading the dataset """
(_, _), (_, _), (test_x, _) = load_dataset(DATASET_PATH)
test_x = test_x [1450: 1455]

print(f"Test: {test_len}/{test_len}")
print("")

""" Load the model """

autoencoder_model = tf.keras.models.load_model(model_path, compile=False) 


""" Prediction & Evaluation """
name_scores = []
i = 1
for x in test_x:
    print(i)
    i += 1
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Reading the image """
    in_img = cv2.imread(x)
    in_img = cv2.resize(in_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    img = in_img.copy()
    img = np.reshape(img, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    img = img.astype('float32') / 255.

    """ Prediction """
    predicted_img = autoencoder_model.predict(img).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    predicted_img = predicted_img * 255
    predicted_img = predicted_img.astype(np.uint8)

    """ Save the results """
    name = name.split('\\')[-1]
    save_image_path = f"{dirct_name}/{name}.png"
    save_results_img_img(in_img, predicted_img, save_image_path)


    """ Calculating the metrics values """
    # Calculate structural similarity
    acc = mean_squared_error(in_img.flatten(), predicted_img.flatten())
    name_scores.append((name, acc))

scores = []
f = open(f"{dirct_name}/score.csv", "w")
f.write("file_name,mse\n")
for (name, acc) in name_scores:
    f.write(f"{name:15s},{acc:1.5f}\n")
    scores.append(acc)

score = np.mean(np.array(scores), axis=0)
print(f'Mean Accuracy: {score}')

