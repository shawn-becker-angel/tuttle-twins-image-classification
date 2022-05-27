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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
# import tensorflow_datasets as tfds

# pip uninstall PIL
# pip install Image

import pandas as pd
import numpy as np

import os

# see https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# pip uninstall keras-preprocessing
# pip install git+https://github.com/keras-team/keras-preprocessing.git
# Import from keras_preprocessing not from keras.preprocessing

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

# pip install matplotlib
import matplotlib
print("matplotlib.version:", matplotlib.__version__)

import matplotlib.pyplot as plt

# verify use of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)

# Constants
IMAGE_SIZE = 128
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE

# batch_size must evenly divide the length of the dataset exactly
# because for the test set, you should sample the images exactly once
BATCH_SIZE = 32
EPOCHS = 50

IMAGE_SOURCE_DIRECTORY = "../src-images/"

train_df = pd.read_csv("../csv-data/train_data.csv",header=None, names=['filename','label'])
test_df = pd.read_csv("../csv-data/test_data.csv",header=None, names=['filename','label'])

# the number of samples used, N, must be 
# divisible by 32, the test batch_size, it must also be
# divisible  by 4, the validation split (but 32 is already a multiple of 4)\

orig_N = len(train_df)
N = (orig_N // BATCH_SIZE) * BATCH_SIZE
print("orig_N:", orig_N, "N:", N)

# truncate the dataframes to N
train_df = train_df[:N]
test_df = test_df[:N]

datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../src-images/",
    x_col="filename",
    y_col="label",
    subset="training",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode="categorical",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

# print("train_generator class_names:", train_generator.class_names)
NUM_CLASSES = len(train_generator.class_indices)

labels_to_ints = train_generator.class_indices
ints_to_labels = {y: x for x, y in labels_to_ints.items()}

valid_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../src-images/",
    x_col="filename",
    y_col="label",
    subset="validation",
    batch_size=BATCH_SIZE, 
    seed=42,
    shuffle=True,
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode="categorical",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="../src-images/",
    x_col="filename",
    y_col=None, # images only
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=False, # not needed for testing
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode=None, # images only
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizers.RMSprop(learning_rate=0.0001),
loss="categorical_crossentropy", metrics=["accuracy"])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
train_valid_history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS)

score = model.evaluate_generator(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict=model.predict_generator(test_generator, steps = len(test_generator.filenames))

y_classes = predict.argmax(axis=-1)
y_filenames = test_generator.filenames
assert len(y_classes) == len(y_filenames)

y_labels = [ints_to_labels[y_classes[i]] for i in range(len(y_classes))]


def list_labels_with_filenames(name, labels, filenames):
    for i in range(len(labels)):
        label = labels[i]
        filename = filenames[i]
        print(f"{name} label:{label} for filename:{filename}")

#     plt.figure(figsize=(10, 10))
#     for images, labels in data_generator.class_names.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(class_names[labels[i]])
#             plt.axis("off")

list_labels_with_filenames("predicted", y_labels, y_filenames,)

def plot_train_valid_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_train_valid_history(train_valid_history)

# def plot_data_generator_images(name, data_generator):
#     '''plot one image for each class'''
#     plt.figure(figsize=(10, 10))
#     for images, labels in data_generator.class_names.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(class_names[labels[i]])
#             plt.axis("off")

# plot_data_generator_images("train", train_generator)

