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
import sys
import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
from matplotlib_utils import \
    plot_idxed_generator_images, \
    plot_idxed_image_files_with_labels, \
    generate_random_plot_idx
    
from history_utils import plot_history, save_history
from shuffle_utils import triple_shuffle_split
from model_file_utils import save_model, load_latest_model, load_model, models_are_equivalent

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("cnn_image_classification")

# verify availability of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float64)
    b = tf.nn.relu(a)
    logger.info(f"a:{a}")
    logger.info(f"b:{b}")

def create_generators(
    csv_data_file, 
    src_imgs_dir,
    label_to_idx_map,
    idx_to_label_map,
    data_splits,
    frame_subsample_rate,
    batch_size,
    target_size,
    plot_random_images=True):

    # read CSV_DATA_FILE, which 
    # has 55352 rows for all tuttle_twins frames from S01E01 to S01E02
    # and has already undergone 10 iterations of shuffling/resampling

    data_df = pd.read_csv(csv_data_file, header=None, dtype=str, names=['filename','label'])
    logger.info(f"data_df.len: {len(data_df)}")

    # counts of each label before frame_subsampling
    y_counts = data_df['label'].value_counts()
    total_label_weights_by_label = max(y_counts)/y_counts

    # keep only 1 out of frame_subsample_rate frames
    if frame_subsample_rate > 1:
        data_df = data_df.iloc[::frame_subsample_rate, :]
        logger.info(f"data_df.len: {len(data_df)} subsampled")

    # the number of samples used, N, must be 
    # divisible by the test batch_size and by the validation split
    N = len(data_df)
    N = (N // batch_size) * batch_size

    # truncate total data_df rows to to N
    data_df = data_df.iloc[:N,:]
    assert len(data_df) == N
    logger.info(f"data_df.len: {len(data_df)} rounded")

    #------------------------------------
    # Split the dataset into X_data and y_data

    X_data = data_df['filename'].to_list()
    y_data = data_df['label'].to_list()

    # do our own label-to-int categorization on the incoming labels
    # so we can use 
    # class_mode 'raw'  and 
    # loss function 'sparse_categorical_crossentropy'
    y_data = [label_to_idx_map[label] for label in y_data]

    # used as class_weights in model.fit(training)
    # convert label_weights by index to label_weights by label
    assert idx_to_label_map is not None
    label_weights_by_idx = {idx:total_label_weights_by_label[idx_to_label_map[idx]] for idx in idx_to_label_map.keys()}

    # validation and test data are shuffled only at start
    # training set is shuffled on each epocn

    # create indices with percentages over N
    train_idx, valid_idx, test_idx = triple_shuffle_split(
        data_size=N,
        data_splits=data_splits, 
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

    logger.info(f"train_df.len: {len(train_df)}")
    logger.info(f"valid_df.len: {len(valid_df)}")
    logger.info(f"test_df.len: {len(test_df)}")

    #------------------------------------
    # Train

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=src_imgs_dir,
        x_col="filename",
        y_col="label",
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        drop_duplicates=False, # not needed
        validate_filenames=False, # not needed
        class_mode="raw", # even though we've done our own categorization?
        interpolation="box",  # prevents antialiasing if subsampling
        target_size=target_size)

    if plot_random_images:
        train_plot_idx = generate_random_plot_idx(train_generator)
        plot_idxed_generator_images(
            name="train", generator=train_generator, 
            plot_idx=train_plot_idx, idx_to_label_map=idx_to_label_map)

    #------------------------------------
    # Valid

    valid_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=src_imgs_dir,
        x_col="filename",
        y_col="label",
        subset=None,
        batch_size=batch_size, 
        seed=42,
        shuffle=False,
        steps_per_epoch=None, # no shuffle if not None
        drop_duplicates=False, # not needed
        validate_filenames=False, # not needed
        class_mode="raw",
        interpolation="box",
        target_size=target_size)

    if plot_random_images:
        valid_plot_idx = generate_random_plot_idx(valid_generator)
        plot_idxed_generator_images(
            "valid", valid_generator, 
            valid_plot_idx, idx_to_label_map)

    #------------------------------------
    # Test

    test_datagen = ImageDataGenerator(rescale=1./255.)

    # image filenames no labels
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=src_imgs_dir,
        x_col="filename",
        y_col=None, # images only
        batch_size=batch_size,
        seed=42,
        shuffle=False, # not needed for testing
        drop_duplicates=False, # not needed
        validate_filenames=False, # not needed
        class_mode=None, # images only
        interpolation="box",
        target_size=target_size)

    test_plot_idx = generate_random_plot_idx(test_generator)

    if plot_random_images:
        plot_idxed_generator_images(
            "test", test_generator, 
            test_plot_idx)

    # image filenames with labels
    true_test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=src_imgs_dir,
        x_col="filename",
        y_col="label",
        batch_size=batch_size,
        seed=42,
        shuffle=False, # not needed for testing
        drop_duplicates=False, # not needed
        validate_filenames=False, # not needed
        class_mode="raw",
        interpolation="box",
        target_size=target_size)

    assert true_test_generator.filenames == test_generator.filenames

    if plot_random_images:
        plot_idxed_generator_images(
            "true_test", true_test_generator, 
            test_plot_idx, idx_to_label_map)

    generators = (train_generator, valid_generator, test_generator, true_test_generator)
    return  (generators, label_weights_by_idx)


def create_model(
    target_size,
    learning_rate, 
    labels): 
    
    assert target_size is not None
    assert learning_rate is not None
    assert labels is not None

    # https://datascience.stackexchange.com/a/24524

    model = Sequential()
    input_shape = (target_size[0], target_size[1], 3) # H,W,C
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
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
    model.add(Dense(len(labels), activation='softmax'))

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
    )

    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])
    
    return model


def fit_model(
    model,
    train_generator,
    valid_generator,
    class_weights_by_idx,
    epochs):
    
    step_size_train=train_generator.n//train_generator.batch_size
    step_size_valid=valid_generator.n//valid_generator.batch_size

    # update the model so that model(train) gradually matches model(valid)
    model.fit(
        train_generator,
        shuffle=True, # shuffle before each epoch
        steps_per_epoch=None, # no shuffle if not None
        validation_data=valid_generator,
        validation_steps=step_size_valid,
        class_weight=class_weights_by_idx,
        epochs=epochs)
    
    return model

def quick_evaluate_model(
    model,
    generator_name,
    generator):
    score = model.evaluate(generator)
    logger.info(f"{generator_name} loss: {score[0]}")
    logger.info(f"{generator_name} accuracy: {score[1]}")

def evaluate_model(
    model,
    valid_generator,
    test_generator,
    true_test_generator,
    idx_to_label_map,
    labels):

    logger.info('Classification Report ')

    # use the model to get y_idx predictions for each image in the test dataset
    Y_test_pred = model.predict(test_generator) # each image has NUM_CLASSES [0..1] prediction probabilities
    y_test_pred = np.argmax(Y_test_pred, axis=1) # each image has a class index with the highest prediction probability

    # This passes 231 == 231
    assert len(test_generator.filenames) == len(true_test_generator.filenames)

    # get the true filenames (X_test_true) and true idx (y_test_true) from the true test dataset
    X_test_true, y_test_true = next(true_test_generator)
    # !! X_test_true and y_test_true have lengths of only 32! 
    # Note that true_test_generator.reset() has no effect.
    # Could it be that its only 32 because only 
    # the first batch of 32 has been run?

    # assert that y_test_pred idx and y_test_true idx have the same lengths
    num_images = len(test_generator.filenames)
    assert len(y_test_pred) == num_images

    # THIS FAILS - 32 != 231
    assert len(y_test_true) == num_images

    # convert idx to labels
    pred_labels = [idx_to_label_map[idx] for idx in y_test_pred]
    true_labels = [idx_to_label_map[idx] for idx in y_test_true]
    
    # Use the test_idx to plot image_files with pred/true_test labels  
    test_idx = generate_random_plot_idx(test_generator)
    pred_v_true_labels = [ f"{pred_labels[i]}[{i}]/{true_labels[i]}[{i}]" for i in range(num_images) ]
    true_filenames = X_test_true
    plot_idxed_imagefiles_with_labels("pred vs true labels", true_filenames, pred_v_true_labels, test_idx)

    # compute and display the confusion matrix of pred vs true labels
    cm = confusion_matrix(y_test_true, y_test_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    logger.info('showing Confusion Matrix of pred vs true labels')
    logger.info(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    plt.show(block=False)

def run_tests():
    logger.info("run_tests() not yet implementated")

def main():
    # usage:
    # python cnn_image_classification.py ( run | test | latest | img-plots-only | <model_dir_path> )

    model = None
    img_plots_only = False
    model_dir_path = None
    if len(sys.argv) > 1:
        argv1 = sys.argv[1]
        if argv1 == 'run':
            pass
        elif argv1 == 'test':
            run_tests()
            return
        elif argv1 == 'latest':
            model = load_latest_model()
            if model is None:
                logger.info("no latest model_dir_path found")
                logger.info("exiting now")
                return
        elif argv1 == 'img-plots-only':
            img_plots_only = True 
        else:
            model_dir_path = argv1
            model = load_model(model_dir_path)
            if model is None:
                raise Exception(f"load_model({model_dir_path}) failed")

    CSV_DATA_FILE = "../csv-data/S01E01-S01E02-data.csv"
    SRC_IMGS_DIR = "../src-images/"
    MODELS_ROOT_DIR = "./models/"

    LABELS = ['Junk', 'Common', 'Uncommon', 'Rare', 'Legendary'] 
    LABEL_TO_IDX_MAP = { 'Junk':0, 'Common':1, 'Uncommon':2, 'Rare':3, 'Legendary':4 }
    IDX_TO_LABEL_MAP = { value: key for key,value in LABEL_TO_IDX_MAP.items()}
 
    # file size
    FILE_IMAGE_HEIGHT = 288
    FILE_IMAGE_WIDTH = 512

    # frame rate and image size
    FRAME_SUBSAMPLE_RATE = 24
    IMAGE_HEIGHT = int(round(FILE_IMAGE_HEIGHT / 2))
    IMAGE_WIDTH = int(round(FILE_IMAGE_WIDTH / 2))
    TARGET_SIZE=(IMAGE_HEIGHT,IMAGE_WIDTH)

    BATCH_SIZE = 32
    EPOCHS = 1
    DATA_SPLITS = {'train_size':0.70, 'valid_size':0.20, 'test_size':0.10}
    LEARNING_RATE = 0.0001
    
    (generators, label_weights_by_idx) = create_generators(
        csv_data_file=CSV_DATA_FILE, 
        src_imgs_dir=SRC_IMGS_DIR,
        label_to_idx_map = LABEL_TO_IDX_MAP,
        idx_to_label_map = IDX_TO_LABEL_MAP,
        data_splits = DATA_SPLITS,
        frame_subsample_rate=FRAME_SUBSAMPLE_RATE,
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
        plot_random_images=img_plots_only)
    
    (train_generator, valid_generator, test_generator, true_test_generator) = generators

    # return early if img_plots_only is True
    if img_plots_only:
        return

    # train/fit the model only if model is None
    if model is None:
        model = create_model(
            target_size=TARGET_SIZE,
            learning_rate=LEARNING_RATE,
            labels=LABELS) 

        model = fit_model(
            model=model,
            train_generator=train_generator,
            valid_generator=valid_generator,
            class_weights_by_idx=label_weights_by_idx,
            epochs=EPOCHS)

        model_dir_path = save_model(MODELS_ROOT_DIR, model)
        logger.info(f"saved model to model_dir_path: {model_dir_path}")
        loaded_model = load_latest_model(MODELS_ROOT_DIR)
        assert models_are_equivalent(model, loaded_model)

    evaluate_model(
        model=model,
        valid_generator=valid_generator,
        test_generator=test_generator,
        true_test_generator=true_test_generator,
        idx_to_label_map=IDX_TO_LABEL_MAP,
        labels=LABELS)

if __name__ == '__main__':
    main()
