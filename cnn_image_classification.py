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
    plot_idxed_generator_images, \
    plot_idxed_image_files_with_labels, \
    generate_random_idx
    
from history_utils import plot_history, save_history
from shuffle_utils import triple_shuffle_split

# verify availability of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float64)
    b = tf.nn.relu(a)
    print("a:", a)
    print("b:", b)


def create_generators(
    csv_data_file=None, 
    src_imgs_dir=None,
    label_to_idx_map=None,
    idx_to_label_map=None,
    data_splits=None,
    frame_subsample_rate=None,
    batch_size=None,
    target_size=None):

    # read CSV_DATA_FILE, which 
    # has 55352 rows for all tuttle_twins frames from S01E01 to S01E02
    # and has already undergone 10 iterations of shuffling/resampling

    data_df = pd.read_csv(csv_data_file, header=None, dtype=str, names=['filename','label'])

    # counts of each label before frame_subsampling
    y_counts = data_df['label'].value_counts()
    total_label_weights_by_label = max(y_counts)/y_counts

    # keep only 1 out of frame_subsample_rate frames
    if frame_subsample_rate > 1:
        data_df = data_df.iloc[::frame_subsample_rate, :]

    # the number of samples used, N, must be 
    # divisible by the test batch_size and by the validation split
    N = len(data_df)
    N = (N // batch_size) * batch_size

    # truncate total data_df rows to to N
    data_df = data_df.iloc[:N,:]
    assert len(data_df) == N

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
    label_weights_by_idx = {i:total_label_weights_by_label[label_to_idx_map[i]] for i in label_to_idx_map.keys()}

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

    train_idx = generate_random_idx(train_generator)
    plot_idxed_generator_images(
        name="train", generator=train_generator, 
        idx=train_idx, idx_to_label_map=idx_to_label_map)

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

    valid_idx = generate_random_idx(valid_generator)
    plot_idxed_generator_images(
        "valid", valid_generator, 
        valid_idx, idx_to_label_map)

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

    test_idx = generate_random_idx(test_generator)
    plot_idxed_generator_images(
        "test", test_generator, 
        test_idx)

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

    plot_idxed_generator_images(
        "true_test", true_test_generator, 
        test_idx, idx_to_label_map)

    generators = (train_generator, valid_generator, test_generator, true_test_generator)
    return  generators, label_weights_by_idx


def create_model(
    target_size=None,
    learning_rate=None, 
    labels=None): 

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

    model.compile(
        optimizers.RMSprop(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"])


def fit_model(
    model=None,
    train_generator=None,
    valid_generator=None,
    class_weights_by_idx=None,
    epochs=None):

    step_size_train=train_generator.n//train_generator.batch_size
    step_size_valid=valid_generator.n//valid_generator.batch_size

    # update the model so that model(train) gradually matches model(valid)
    train_valid_history = model.fit_generator(
        generator=train_generator,
        shuffle=True, # shuffle before each epoch
        steps_per_epoch=None, # no shuffle if not None
        validation_data=valid_generator,
        validation_steps=step_size_valid,
        class_weight=class_weights_by_idx,
        epochs=epochs)

def quick_evaluate_model(
    model=None,
    generator_name=None,
    generator=None):
    score = model.evaluate(generator)
    print(generator_name, 'loss:', score[0])
    print(generator_name, 'accuracy:', score[1])

def save_model(models_root_dir=None, model=None):
    dt = datetime.datetime.utcnow().isoformat()
    model_dir_path = os.path.join(models_root_dir, f"model-{dt}")
    model.save(model_dir_path)

def find_latest_model_dir_path(models_root_dir=None):
    '''find the latest model_dir_path under models_root_dir'''
    def sort_dict_by_value(d, reverse = False):
        return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

    only_model_dirs = [f for f in os.listdir(models_root_dir) if isdir(join(models_root_dir, f) and f.startswith("model-") )]
    if len(only_model_dirs) == 0:
        return None
    ctimes = {f: os.path.getctime(f) for f in only_model_dirs}
    ctimes_r = sort_dict_by_value(ctimes, reverse = True)
    return ctimes_r[0].key()

def load_latest_model(models_root_dir=None):
    model_dir_path = find_latest_model_dir_path(models_root_dir)
    if model_dir_path is not None:
        return keras.models.load_model(model_dir_path)
    return None


def evaluate_model(
    model=None,
    valid_generator=None,
    test_generator=None,
    true_test_generator=None,
    idx_to_label_map=None,
    labels=None):

    test_idx = generate_random_idx(test_generator)

    # Confusion Matrix and Classification Report
    Y_valid_pred = model.predict(valid_generator)
    y_valid_pred = np.argmax(Y_valid_pred, axis=1)

    print('Confusion Matrix')
    print(confusion_matrix(labels, y_valid_pred))

    print('Classification Report')
    target_names = [idx_to_label_map[idx] for idx in np.unique(y_valid_pred)]
    assert set(target_names).issubset(set(labels))
    assert set(y_valid_pred).issubset(set(labels))

    # use the model to get class predictions for each image in the test dataset
    Y_test_pred = model.predict(test_generator) # each image has NUM_CLASSES [0..1] prediction probabilities
    y_test_pred = np.argmax(Y_test_pred, axis=1) # each image has a class index with the highest prediction probability
    assert len(Y_test_pred) == len(y_test_pred)

    # get the actual Images (x_test_true) and actual labels (y_test_true) from the test dataset
    x_test_true, y_test_true = next(true_test_generator)
    assert len(x_test_true) == len(y_test_true)

    filenames = test_generator.filenames
    num_images = len(filenames)
    assert len(y_test_pred) == num_images

    # Plot image_files with pred/true_test labels  
    pred_v_pred_true_labels = [ f"{y_test_pred[i]}[{i}]/{y_test_true[i]}[{i}]" for i in range(num_images) ]
    pred_v_true_filenames = x_test_true
    plot_idxed_imagefiles_with_labels("pred vs true labels", pred_v_true_filenames, pred_v_true_labels, test_idx)

    # Plot the confusion matrix  of pred vs true
    pred_true_display_labels = labels
    cm = confusion_matrix(y_test_pred, y_test_true)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=pred_true_display_labels)
    disp.plot(cmap=plt.cm.Blues)
    print(f"Showing confusion matrix pred vs true labels")
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    plt.show(block=False)

def main():

    CSV_DATA_FILE = "../csv-data/S01E01-S01E02-data.csv"
    SRC_IMGS_DIR = "../all-src-images/"
    MODELS_DIR = "./models/"

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
    EPOCHS = 50
    DATA_SPLITS = {'train_size':0.70, 'valid_size':0.20, 'test_size':0.10}
    LEARNING_RATE = 0.0001

    generators, label_weights_by_idx = create_generators(
        csv_data_file=CSV_DATA_FILE, 
        src_imgs_dir=SRC_IMGS_DIR,
        label_to_idx_map = LABEL_TO_IDX_MAP,
        idx_to_label_map = IDX_TO_LABEL_MAP,
        data_splits = DATA_SPLITS,
        frame_subsample_rate=FRAME_SUBSAMPLE_RATE,
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE)

    (train_generator, valid_generator, test_generator, true_test_generator) = generators

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

    #----------- save/load test ----------------#
    pre_saved_score = quick_evaluate_model(
        model, "valid_generator", valid_generator)

    save_model( models_root_dir=MODELS_ROOT_DIR, model=model)
    
    saved_model = load_latest_model(models_root_dir)

    saved_score = quick_evaluate_model(
        saved_model, "valid_generator", valid_generator)

    assert saved_score == pre_saved_score
    #----------- save/load test ----------------#

    evaluate_model(
        model=model,
        valid_generator=valid_generator,
        test_generator=test_generator,
        true_test_generator=true_test_generator,
        idx_to_label_map=IDX_TO_LABEL_MAP,
        labels=LABELS)

if __name__ == '__main__':
    main()
