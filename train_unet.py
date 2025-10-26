import os
import time
import numpy as np
import tensorflow as tf
import segmentation_models as sm

from dataset_handler import *
from image_preprocess import *
from global_parameters import *
from encoder_decoder_models import build_unet, second_to_time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7400)])
  except RuntimeError as e:
    print(e)

USE_TRANSFER_LEARNING = True
ENCODER_LAYERS_COUNT = 35

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("files")

""" Hyperparameters """
train_len, valid_len = 10000, 1000

""" Loading the dataset """
(train_x, train_y), (valid_x, valid_y), (_, _) = load_dataset(DATASET_PATH)

if len(train_len) <= train_x and len(valid_len) <= valid_len:
    (train_x, train_y) = (train_x[0: train_len], train_y[0: train_len])
    (valid_x, valid_y) = (valid_x[0: valid_len], valid_y[0: valid_len])

train_len, valid_len = len(train_x), len(valid_x)
if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
    exit(-1)

""" Dataset Pipeline """
train_ds = tf_dataset(train_x, train_y)
valid_ds = tf_dataset(valid_x, valid_y)

print(f"Train: {len(train_x)} - Valid: {len(valid_x)}")
print("")

""" Paths """
model_name = 'UNet'
using_transfer_learning = 'transfer_learning' if USE_TRANSFER_LEARNING else 'not_transfer_learning'
model_info = f'{model_name}_{train_len}imgs_{EPOCHS_COUNT}epochs_{using_transfer_learning}'
model_path = os.path.join("files", f"model_{model_info}.h5")
csv_path = os.path.join("files", f"data_{model_info}.csv")
info_path = os.path.join("files", f"info_{model_info}.txt")

""" Model """
unet_model = build_unet(INPUT_SHAPE, NUM_CLASSES)

if USE_TRANSFER_LEARNING:
    autoencoder_model_name = 'model_autoencoder_20000imgs_8epochs.h5'
    autoencoder_model = tf.keras.models.load_model(os.path.join("files", autoencoder_model_name), compile=False)
    
    if len(unet_model.layers[: ENCODER_LAYERS_COUNT]) == len(autoencoder_model.layers[: ENCODER_LAYERS_COUNT]):
        for l1, l2 in zip(unet_model.layers[: ENCODER_LAYERS_COUNT], autoencoder_model.layers[: ENCODER_LAYERS_COUNT]):
            l1.set_weights(l2.get_weights())
    else:
        print('unet and autoencoders must have same layers.')
        exit(-1)

unet_model.compile(
    loss=[sm.losses.categorical_focal_jaccard_loss],
    metrics=[sm.metrics.iou_score],
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
)

""" Training """
print(f'using transfer learning = {USE_TRANSFER_LEARNING}')

start_time = time.time()
unet_model.fit(train_ds,
               validation_data=valid_ds,
               epochs=EPOCHS_COUNT,
               callbacks=get_callbacks())

writable_info = [
    f'Model Name: [{model_name}]',
    f'Use Tansfer Learning? [{USE_TRANSFER_LEARNING}]',
    f'Images Size: [{str(INPUT_SHAPE)}]',
    f'Training Dataset Length: [{train_len}]',
    f'Validation Dataset Length: [{valid_len}]',
    f'Number Of Classes: [{NUM_CLASSES}]',
    f'Classes Names: [{str(CLASSES)}]',
    f'Batch Size: [{BATCH_SIZE}]',
    f'Epochs Count: [{EPOCHS_COUNT}]',
    f'Start Learnin Rate: [{LEARNING_RATE}]',
    f'Layers Count: [{len(unet_model.layers)}]',
    f'Execute Time: [{second_to_time((time.time() - start_time))}]'
    ]

f = open(info_path, "w")
f.writelines(writable_info)
f.close()

