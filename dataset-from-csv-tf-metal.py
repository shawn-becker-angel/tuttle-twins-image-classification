# see https://towardsdatascience.com/accelerated-tensorflow-model-training-on-intel-mac-gpus-aa6ee691f894
# by Bryan M. Li 
# Published in Towards Data Science - Oct 26, 2021
#
# special setup instructions for macos Monterey 12.3.1
#
# python3.8 -m venv venv-tf-metal
# source venv-tf-metal/bin/activate
# python3.8 -m pip install --upgrade pip
# SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos tensorflow-metal
# pip install tensorflow_datasets

from sklearn.semi_supervised import LabelSpreading
import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
import os

# verify using GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)

# Constants
IMAGE_SIZE = 128
NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE

IMAGE_SOURCE_DIRECTORY = "../src-images/"
if not os.path.isdir(IMAGE_SOURCE_DIRECTORY):
    raise Exception(f"no such directory: {IMAGE_SOURCE_DIRECTORY}")



# helper functions

def read_image(file_name, label):
    image = tf.io.read_file(
        IMAGE_SOURCE_DIRECTORY + file_name)
    image = tf.image.decode_jpeg(
        image, 
        channels=3,
        ratio=4)                # downscaling_ratio to get us close to final size
    image = tf.image.resize(
        image, size=(128, 128),
        preserve_aspect_ratio=False, 
        antialias=True)
    return image, label

def augment_image(image, label):
    return image, label

def normalize_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255. # normalize
    return image, label

def get_dataset_partitions_tf(
    ds, ds_size, train_split=0.8, test_split=0.1, pred_split=0.1, 
    shuffle=True, shuffle_size=10000):
    
    assert (train_split + pred_split + test_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(test_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# read the train data file into train_df
train_csv_file =  "../csv-data/train_data.csv"
if not os.path.isfile(train_csv_file):
    raise Exception(f"no such file: {train_csv_file}")
train_df = pd.read_csv(train_csv_file,header=None, names=['file_name','label'])
print("train_df.shape:", train_df.shape)

# read the test data file into test_df
test_csv_file =  "../csv-data/test_data.csv"
if not os.path.isfile(test_csv_file):
    raise Exception(f"no such file: {test_csv_file}")
test_df = pd.read_csv(test_csv_file,header=None, names=['file_name','label'])
print("test_df.shape:", test_df.shape)

# combine train_df and test_df into a single df
df = pd.concat([train_df,test_df],axis=0)
print("df.shape:", df.shape)

# file_names and labels vectors
file_names = df['file_name'].values
labels = df['label'].values
N = len(file_names)
assert len(labels == N)

# create ds_train, the train dataset
ds_train = tf.data.Dataset.from_tensor_slices((file_names, labels))

ds_train = ds_train.map(read_image).map(augment_image).batch(16)

# split out ds_test and ds_pred from ds_train
(ds_train, ds_test, ds_pred) = get_dataset_partitions_tf(
    ds_train, N, train_split=0.8, test_split=0.1, pred_split=0.1, 
    shuffle=True, shuffle_size=10000)

# prep ds_train
ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# prep ds_test
ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)

print('\ncreate the model')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES)
])

print('\ncompile the model')
model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

print("\nnow fit model to ds_train and ds_test")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)


# for epoch in range(EPOCHS):
#     for x, y in ds_train:
#         # train here
#         pass

