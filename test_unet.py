import os
import numpy as np
import tensorflow as tf

from image_preprocess import *
from sklearn.metrics import f1_score, jaccard_score
from dataset_handler import create_dir, load_dataset, save_results_img_mask_mask_2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Hyperparameters """
test_len = 50

""" Paths """
model_name = 'model_Attention_UNet_Conv2D_10000imgs_3epochs_transfer_learning1.h5'
model_path = os.path.join("files", model_name)

""" Directory for storing files """
dirc_name = f'{model_name}_results'
create_dir(dirc_name)

test_len = 1000

""" Loading the dataset """
(_, _), (_, _), (test_x, test_y) = load_dataset(DATASET_PATH)
(test_x, test_y) = (test_x [0: test_len], test_y[0: test_len])

print(f"Test: {test_len}/{test_len}")
print("")

""" Load the model """
model = tf.keras.models.load_model(model_path, compile=False)

""" Prediction & Evaluation """
ears_score = []
i = 1
mean_score_all = []
for x, y in zip(test_x, test_y):
    print(i)
    i += 1
    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ Reading the image and mask"""
    image, mask = read_image_mask(x, y)
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
    
    save_results_img_mask_mask_2(image_x, mask, pred, save_image_path)

    """ Flatten the array """

    mask = mask.flatten()
    pred = pred.flatten()

    labels = [i for i in range(NUM_CLASSES)]

    """ Calculating the metrics values """

    f1_value = f1_score(mask, pred, labels=labels, average=None, zero_division=0)
    jac_value = jaccard_score(mask, pred, labels=labels, average=None, zero_division=0)
    if f1_value[5] > 0.0:
        ears_score.append([f1_value[5], jac_value[5]])

    mean_score_all.append([f1_value, jac_value])


mean_score_all = np.mean(np.array(mean_score_all), axis=0)
mean_score_all[:, 5] = np.mean(np.array(ears_score), axis=0)

f = open(f"{dirc_name}/scores.csv", "w")
f.write("Class,F1,iou\n")

l = ["Class", "F1", "iou"]
print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
print("-" * 35)

for i in range(NUM_CLASSES):
    class_name = CLASSES[i]
    f1 = mean_score_all[0, i]
    jac = mean_score_all[1, i]
    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

print("-" * 35)
class_mean = np.mean(mean_score_all, axis=-1)
class_name = "Mean"

f1 = class_mean[0]
jac = class_mean[1]

dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
print(dstr)
f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

f.close()