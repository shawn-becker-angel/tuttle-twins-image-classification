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
# pip uninstall PIL
# pip install Image
# pip uninstall keras-preprocessing
# pip install git+https://github.com/keras-team/keras-preprocessing.git

# Import from keras_preprocessing not from keras.preprocessing
# see https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

import pandas as pd
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ShuffleSplit

from matplotlib_utils import \
    plot_random_generator_images_with_labels, \
    plot_random_imagefiles_with_labels, \
    plot_random_generator_images_no_labels
    
from history import plot_history, save_history

# verify availability of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)


# batch_size must evenly divide the length of the dataset exactly
# because for the test set, you should sample the images exactly once
BATCH_SIZE = 32
EPOCHS = 50

CSV_DATA_FILE = "../csv-data/S01E01-S01E02-data.csv"
IMAGE_SOURCE_DIRECTORY = "../src-images/"

# Constants
FILE_IMAGE_HEIGHT = 288
FILE_IMAGE_WIDTH = 512

IMAGE_HEIGHT = FILE_IMAGE_HEIGHT / 2
IMAGE_WIDTH = FILE_IMAGE_WIDTH / 2

# the number of samples used, N, must be 
# divisible by 32, the test batch_size, it must also be
# divisible  by 4, the validation split (but 32 is already a multiple of 4)\
N = 1024
assert (N // BATCH_SIZE) * BATCH_SIZE == N

# read CSV_DATA_FILE, which 
# has 55352 rows for all tuttle_twins frames in S01E01 and S01E02
# and has undergone 10 iterations of shuffling/resampling

data_df = pd.read_csv(CSV_DATA_FILE,header=None, names=['filename','label'], nrows=N)
assert len(data_df) == N

#------------------------------------
# Split the dataset

X_data = data_df['filename'].to_list()
y_data = data_df['label'].to_list();

# do our own label-to-int categorization on the incoming labels
label_map = { 'Junk':0, 'Common':1, 'Uncommon':2, 'Rare':3, 'Legendary':4 }
y_data = [label_map[label] for label in y_data]

# split data into .9 train and .1 test
train_indices, test_indicies = ShuffleSplit(n_splits=10, train_size=.9, test_size=.1, random_state=123) 
X_train = X_data[train_indices]
y_train = y_data[train_indices]
X_test = X_data[test_indicies]
y_test = y_data[test_indicies]

# split train into .7 train and .3 valid
train_train_indices, train_valid_indicies = ShuffleSplit(train_size=.7, test_size=.3, random_state=123) 
X_train = X_train[train_train_indices]
y_train = y_train[train_train_indices]
X_valid = X_train[train_valid_indicies]
y_valid = y_train[train_valid_indicies]

# yields .9x.7 train + .9x.3 valid + 0.1 test = 1.0 data

# concat X and y arrays horizontally to create dataframes
train_df = pd.concat([X_train,y_train], axis=1)
valid_df = pd.concat([X_valid,y_valid], axis=1)
test_df = pd.concat([X_test,y_test], axis=1)
assert len(train_df) + len(valid_df) + len(test_df) == N

#------------------------------------
# Train

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="../src-images/",
    x_col="filename",
    y_col="label",
    subset=None,
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode=None, # we did our own categorization
    interpolation="box",  # prevents antialiasing if subsampling
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

# print("train_generator class_names:", train_generator.class_names)
CLASSES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASSES)

plot_random_generator_images_with_labels("train", train_generator)

#------------------------------------
# Validation

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
    interpolation="box",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

assert (len(valid_generator.filenames) + len(train_generator.filenames)) == len(train_df)

plot_random_generator_images_with_labels("valid", valid_generator)

#------------------------------------
# Test

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
    interpolation="box",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

true_test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="../src-images/",
    x_col="filename",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=False, # not needed for testing
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode="categorical", # images only
    interpolation="box",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

assert true_test_generator.filenames == test_generator.filenames

plot_random_generator_images_no_labels("test", test_generator)
plot_random_generator_images_with_labels("true_test", true_test_generator)

# We need to shuffle the training data for each epoch
# https://datascience.stackexchange.com/a/24524

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

model.compile(
    optimizers.RMSprop(learning_rate=0.0001),
    loss="categorical_crossentropy", 
    metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# update the model so that model(train) gradually matches model(valid)
train_valid_history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS)

score = model.evaluate_generator(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save_history(filename="train-valid-history", history=train_valid_history)
# plot_history(name="train-valid-history", history=train_valid_history)

# Confusion Matrix and Classification Report
Y_valid_pred = model.predict_generator(valid_generator)
y_valid_pred = np.argmax(Y_valid_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(valid_generator.classes, y_valid_pred))

print('Classification Report')
index_to_class = {value:key for key, value in valid_generator.class_indices.items()}
target_names = [index_to_class[idx] for idx in np.unique(y_valid_pred)]

assert len(y_valid_pred) == len(valid_generator.classes)

pred_classes_set = set(y_valid_pred)
valid_classes_set = set(valid_generator.classes)
if pred_classes_set != valid_classes_set:
    print(f"INFO: pred classes: {pred_classes_set} != valid_classes_set: {valid_classes_set}")

# use the model to get class predictions for each image in the test dataset
Y_test_pred = model.predict_generator(test_generator) # each image has NUM_CLASSES [0..1] prediction probabilities
y_test_pred = np.argmax(Y_test_pred, axis=1) # each image has a class index with the highest prediction probability
assert len(Y_test_pred) == len(y_test_pred)

# get the actual Images (x_test_true) and actual classes (y_test_true) from the test dataset
x_test_true, y_test_true = next(true_test_generator)
assert len(x_test_true) == len(y_test_true)

# assert len(test_generator.filenames) == len(x_test_true)
# assert set(test_generator.filenames) == set(x_test_true)
# assert test_generator.filenames == x_test_true

filenames = test_generator.filenames
num_images = len(filenames)
assert len(y_test_pred) == num_images

# assert len(y_test_true) == num_images

# Plot image_files with pred/true_test classes (aka labels) 
pred_v_true_labels = [ f"{y_test_pred[i]}[{i}]/{y_test_true[i]}[{i}]" for i in range(num_images) ]
pred_v_true_filenames = x_test_true
plot_random_imagefiles_with_labels("test dataset pred vs true classes", pred_v_true_filenames, pred_v_true_labels)

# Plot the confusion matrix  of pred vs true
pred_true_display_labels = test_generator.class_indices.keys()
cm = confusion_matrix(y_test_pred, y_test_true)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=pred_true_display_labels)
disp.plot(cmap=plt.cm.Blues)
print(f"Showing pred vs true labels")
print(f"Click key or mouse in window to close.")
plt.waitforbuttonpress()
plt.close("all")
plt.show(block=False)

print("done")


