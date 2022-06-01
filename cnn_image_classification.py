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

from matplotlib import pyplot as plt
from matplotlib_utils import \
    plot_random_generator_images_with_labels, \
    plot_random_imagefiles_with_labels, \
    plot_random_generator_images_no_labels
    
from history_utils import plot_history, save_history
from shuffle_utils import triple_shuffle_split

# verify availability of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float64)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)

CSV_DATA_FILE = "../csv-data/S01E01-S01E02-data.csv"
IMAGE_SOURCE_DIRECTORY = "../src-images/"

# file size
FILE_IMAGE_HEIGHT = 288
FILE_IMAGE_WIDTH = 512

# target image size
IMAGE_HEIGHT = int(round(FILE_IMAGE_HEIGHT / 2))
IMAGE_WIDTH = int(round(FILE_IMAGE_WIDTH / 2))

# read CSV_DATA_FILE, which 
# has 55352 rows for all tuttle_twins frames from S01E01 to S01E02
# and has already undergone 10 iterations of shuffling/resampling

data_df = pd.read_csv(CSV_DATA_FILE,header=None, names=['filename','label'])

# counts of each label
y_counts = data_df['label'].value_counts()
label_weights = max(y_counts)/y_counts

# keep only 1 out of 24 frames
data_df = data_df.iloc[::24, :]

# batch_size must evenly divide the length of the dataset exactly
# because for the test set, you should sample the images exactly once
BATCH_SIZE = 32
EPOCHS = 50

# the number of samples used, N, must be 
# divisible by 32, the test batch_size, it must also be
# divisible  by 4, the validation split (but 32 is already a multiple of 4)\
N = len(data_df)
N = (N // BATCH_SIZE) * BATCH_SIZE

# truncate total data_df rows to to N
data_df = data_df.iloc[:N,:]
assert len(data_df) == N

#------------------------------------
# Split the dataset into X_data and y_data

X_data = data_df['filename'].to_list()
y_data = data_df['label'].to_list()

label_map_int = { 'Junk':0, 'Common':1, 'Uncommon':2, 'Rare':3, 'Legendary':4 }
int_map_label = { value: key for key,value in label_map_int.items()}
NUM_CLASSES = len(label_map_int)
assert NUM_CLASSES == 5

# do our own label-to-int categorization on the incoming labels
y_data = [label_map_int[label] for label in y_data]

# used as class_weights in model.fit(training)
label_weights = {i:label_weights[int_map_label[i]] for i in int_map_label.keys()}

# validation and test data are shuffled only at start
# training set is shuffled on each epocn

# define data splits
train_size = 0.70
valid_size = 0.20
test_size =  0.10

# create indices with percentages over N
train_idx, valid_idx, test_idx = triple_shuffle_split(
    data_size=N, 
    train_size=train_size, 
    valid_size=valid_size, 
    test_size=test_size, 
    random_state=123)

# prepare for indexing
X_data = np.array(X_data)
y_data = np.array(y_data)

# apply indices and name each series
X_train = pd.Series(X_data[train_idx], name='filename')
y_train = pd.Series(y_data[train_idx], name='label', dtype='float64')

X_valid = pd.Series(X_data[valid_idx], name='filename')
y_valid = pd.Series(y_data[valid_idx], name='label', dtype='float64')

X_test = pd.Series(X_data[test_idx], name='filename')
y_test = pd.Series(y_data[test_idx], name='label', dtype='float64')

# concat X and y series horizontally to create dataframes
train_df = pd.concat([X_train, y_train], axis=1)
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
    shuffle=False,
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode="raw", # even though we've done our own categorization?
    interpolation="box",  # prevents antialiasing if subsampling
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

plot_random_generator_images_with_labels("train", train_generator)

#------------------------------------
# Valid

valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory="../src-images/",
    x_col="filename",
    y_col="label",
    subset=None,
    batch_size=BATCH_SIZE, 
    seed=42,
    shuffle=False,
    steps_per_epoch=None, # no shuffle if not None
    class_weight=label_weights,
    drop_duplicates=False, # not needed
    validate_filenames=False, # not needed
    class_mode=None,
    interpolation="box",
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH))

plot_random_generator_images_with_labels("valid", valid_generator)

#------------------------------------
# Test

test_datagen = ImageDataGenerator(rescale=1./255.)

# image filenames only
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

# image filenames with labels
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
    class_mode=None, # images only
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
    shuffle=True, # shuffle before each epoch
    steps_per_epoch=None, # no shuffle if not None
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    class_weight=label_weights,
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


