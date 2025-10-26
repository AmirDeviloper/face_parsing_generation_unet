import time
import tensorflow as tf

from dataset_handler import *
from image_preprocess import *
from global_parameters import *
from encoder_decoder_models import build_autoencoder, second_to_time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Directory for storing files """
create_dir("files")

""" Hyperparameters """
train_len, valid_len = 10000, 1000


""" Loading the dataset """
(train_x, _), (valid_x, _), (_, _) = load_dataset(DATASET_PATH)

if len(train_len) <= train_x and len(valid_len) <= valid_len:
    (train_x, train_y) = train_x[0: train_len]
    (valid_x, valid_y) = valid_x[0: valid_len]

train_len, valid_len = len(train_x), len(valid_x)
if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
    exit(-1)

""" Dataset Pipeline """
train_ds = tf_dataset(train_x, train_x, for_autoencoder=True)
valid_ds = tf_dataset(valid_x, valid_x, for_autoencoder=True)

print(f"Train: {len(train_x)} - Valid: {len(valid_x)}")
print("")


""" Paths """
model_name = 'AutoEncoder'
model_info = f'{model_name}_{train_len}imgs_{EPOCHS_COUNT}epochs'
model_path = os.path.join("files", f"model_{model_info}.h5")
csv_path = os.path.join("files", f"data_{model_info}.csv")
info_path = os.path.join("files", f"info_{model_info}.txt")

""" Model """
autoencoder_model = build_autoencoder(INPUT_SHAPE)
autoencoder_model.compile(
    loss='mean_squared_error',
    metrics='accuracy',
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
)
    
""" Training """
start_time = time.time()
autoencoder_model.fit(train_ds,
                      validation_data=valid_ds,
                      epochs=EPOCHS_COUNT,
                      callbacks=get_callbacks(model_path, csv_path),
                      shuffle=True)


writable_info = [
    f'Model Name: [{model_name}]',
    f'Images Size: [{str(INPUT_SHAPE)}]',
    f'Training Dataset Length: [{train_len}]',
    f'Validation Dataset Length: [{valid_len}]',
    f'Batch Size: [{BATCH_SIZE}]',
    f'Epochs Count: [{EPOCHS_COUNT}]',
    f'Start Learnin Rate: [{LEARNING_RATE}]',
    f'Execute Time: [{second_to_time((time.time() - start_time))}]'
    ]

f = open(info_path, "w")
f.writelines(writable_info)
f.close()



