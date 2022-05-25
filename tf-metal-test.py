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
import tensorflow_datasets as tfds
import pandas as pd
import os

tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)

BATCH_SIZE = 64
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE

# # load MNIST
print('\ndownload the mnist dataset')
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)

print('\ncreate and compile model')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

print("\nnow fit model to ds_train and ds_test")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)
