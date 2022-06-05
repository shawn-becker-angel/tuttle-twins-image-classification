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

import logging
import model_file_utils
from shuffle_utils import triple_shuffle_split
from history_utils import plot_history, save_history
from matplotlib_utils import \
    plot_idxed_generator_images, \
    plot_idxed_image_files_with_labels, \
    generate_random_plot_idx, \
    plot_model_fit_history
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import sklearn
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
import tensorflow as tf
import numpy as np
import os
import sys
import datetime
import pandas as pd

print("pd",pd.__version__)
print("np",np.__version__)
print("tf",tf.__version__)
print("keras",keras.__version__)
print("sklearn",sklearn.__version__)
print("matplotlib",matplotlib.__version__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cnn_image_classification")


# verify availability of GPU
tf.config.list_physical_devices()
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float64)
    b = tf.nn.relu(a)
    logger.info(f"a:{a}")
    logger.info(f"b:{b}")

#------------ <create_generators> --------------#


def create_generators(
    csv_data_file, 
    src_imgs_dir,
    label_to_idx_map,
    idx_to_label_map,
    data_splits,
    frame_subsample_rate,
    batch_size,
    target_size,
    plot_random_images):

    # read CSV_DATA_FILE, which 
    # has 55352 rows for all tuttle_twins frames from S01E01 to S01E02
    # and has already undergone 10 iterations of shuffling/resampling

    data_df = pd.read_csv(csv_data_file, header=None,
                          dtype=str, names=['filename', 'label'])
    logger.info(f"data_df.len: {len(data_df)}")

    # counts and weights of each label before frame_subsampling
    y_counts = data_df['label'].value_counts()
    total_label_weights_by_label = max(y_counts) / y_counts

    # keep only 1 out of frame_subsample_rate frames
    if frame_subsample_rate > 1:
        data_df = data_df.iloc[::frame_subsample_rate, :]
        logger.info(f"data_df.len: {len(data_df)} subsampled")

    # the number of samples used, N, must be 
    # divisible by the test batch_size and by the validation splits
    N = len(data_df)
    N = (N // batch_size) * batch_size

    # truncate total data_df rows to to N
    data_df = data_df.iloc[:N, :]
    assert len(data_df) == N
    logger.info(f"data_df.len: {len(data_df)} rounded")

    # ------------------------------------
    # Split the dataset into X_data and y_data

    X_data = data_df['filename'].to_list()
    y_data = data_df['label'].to_list()

    # convert X_data filenames to absolute paths
    X_data = [os.path.join(src_imgs_dir, f) for f in X_data]

    # do our own label-to-int categorization on the incoming labels
    # so we can use class_mode 'raw'  and  
    # loss function 'sparse_categorical_crossentropy'
    y_data = [label_to_idx_map[label] for label in y_data]

    # used as class_weights in model.fit(training)
    # convert label_weights by index to label_weights by label
    label_weights_by_idx = {
        idx: total_label_weights_by_label[idx_to_label_map[idx]] for idx in idx_to_label_map.keys()}

    # create indices with percentages over N
    train_idx, valid_idx, test_idx = triple_shuffle_split(
        data_size=N,
        data_splits=data_splits, 
        random_state=123)

    # truncate train_idx to be a multiple of batch_size
    N_train = len(train_idx) // batch_size * batch_size
    train_idx = train_idx[:N_train]

    # prepare for triple_shuffle_split indexing
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # apply indices and name each series
    X_train = pd.Series(X_data[train_idx], name='filename')
    y_train = pd.Series(y_data[train_idx], name='label', dtype='float64')
        
    X_valid = pd.Series(X_data[valid_idx], name='filename')
    y_valid = pd.Series(y_data[valid_idx], name='label', dtype='float64')

    X_test = pd.Series(X_data[test_idx], name='filename')
    y_test = pd.Series(y_data[test_idx], name='label', dtype='float64')

    # true is a duplicate of test
    X_true = pd.Series(X_data[test_idx], name='filename')
    y_true = pd.Series(y_data[test_idx], name='label', dtype='float64')

    # concat X and y series horizontally to create dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    true_df = pd.concat([X_true, y_true], axis=1)

    assert len(train_df) + len(valid_df) + len(test_df) <= N

    logger.info(f"train_df.len: {len(train_df)}")
    logger.info(f"valid_df.len: {len(valid_df)}")
    logger.info(f"test_df.len: {len(test_df)}")

    # train_generator -------------------------
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,  # filenames are already paths
        x_col="filename",
        y_col="label",
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        validate_filenames=False,  # not needed
        class_mode="raw",  # even though we've done our own categorization?
        interpolation="box",  # prevents antialiasing if subsampling
        target_size=target_size)

    if plot_random_images:
        train_plot_idx = generate_random_plot_idx(train_generator)
        plot_idxed_generator_images(
            name="train", generator=train_generator, 
            plot_idx=train_plot_idx, idx_to_label_map=idx_to_label_map)

    # valid_generator ------------------------------
    valid_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=None,  # filenames are already paths
        x_col="filename",
        y_col="label",
        subset=None,
        batch_size=batch_size, 
        seed=42,
        shuffle=False,
        steps_per_epoch=None,  # no shuffle if not None
        validate_filenames=False,  # not needed
        class_mode="raw",
        interpolation="box",
        target_size=target_size)

    if plot_random_images:
        valid_plot_idx = generate_random_plot_idx(valid_generator)
        plot_idxed_generator_images(
            "valid", valid_generator, 
            valid_plot_idx, idx_to_label_map)

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    # test_generator ----------------------------
    # image filenames no labels
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,  # filenames are already paths
        x_col="filename",
        y_col=None,  # images only
        batch_size=test_df.shape[0],  # all available frames
        seed=42,
        shuffle=False,  # not needed for testing
        validate_filenames=False,  # not needed
        class_mode=None,  # images only
        interpolation="box",
        target_size=target_size)

    test_plot_idx = generate_random_plot_idx(test_generator)

    re_X_test = test_generator.next()

    if plot_random_images:
        plot_idxed_generator_images(
            "test", test_generator, 
            test_plot_idx)

    # true_generator -------------------------
    # image filenames with labels
    true_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,  # filenames are already paths
        x_col="filename",
        y_col="label",
        batch_size=test_df.shape[0],  # all frames
        seed=42,
        shuffle=False,  # not needed for testing
        validate_filenames=False,  # not needed
        class_mode="raw",
        interpolation="box",
        target_size=target_size)

    re_X_true, re_y_true = true_generator.next()

    assert np.array_equal(true_generator.filenames, test_generator.filenames)
    assert np.array_equal(re_X_true, re_X_test)
    assert np.array_equal(re_y_true, test_df['label'])

    if plot_random_images:
        plot_idxed_generator_images(
            "true", true_generator, 
            test_plot_idx, idx_to_label_map)

    # batch_cnt = 0
    # for batch_idx, (x, y) in enumerate(train_generator):
    #     batch_cnt += 1
    # print(batch_cnt)

    generators = (train_generator, valid_generator, test_generator)
    return (generators, true_df, label_weights_by_idx)

#------------ </create_generators> --------------#

#------------ <create_model> --------------#


def create_model(
    target_size,
    dropout1,
    dropout2,
    labels): 

    # https://datascience.stackexchange.com/a/24524

    model = Sequential()
    input_shape = (target_size[0], target_size[1], 3)  # H,W,C
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(len(labels), activation='softmax'))

    return model

#------------ </create_model> --------------#

#------------ <fit_model> --------------#


def fit_model(
    model,
    train_generator,
    valid_generator,
    label_weights_by_idx,
    learning_rate,
    epochs):

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
        metrics=[ # https://keras.io/api/metrics/
            "Accuracy",
                # "CategoricalAccuracy",
            # "SparseCategoricalAccuracy",
                # "TopKCategoricalAccuracy",
            # "SparseTopKCategoricalAccuracy",
                # "BinaryCrossentropy",
                # "CategoricalCrossentropy",
            # "SparseCategoricalCrossentropy",
                # "KLDivergence",
                # "Poisson",
                # "MeanSquaredError",
            # "RootMeanSquaredError",
                # "MeanAbsoluteError",
                # "MeanAbsolutePercentageError",
                # "MeanSquaredLogarithmicError",
                # "CosineSimilarity",
                # "LogCoshError",
            # "AUC",
            # "Precision",
                # "Recall",
                # "TruePositives",
                # "TrueNegatives",
                # "FalsePositives",
                # "FalseNegatives",
                # "PrecisionAtRecall",
                # "SensitivityAtSpecificity",
                # "SpecificityAtSensitivity",
        ])

    logging.info(f"fitting model in {epochs} epochs")

    step_size_valid = valid_generator.n // valid_generator.batch_size

    # step_size_train is not used because train is shuffled before each epoch
    # step_size_train=train_generator.n//train_generator.batch_size 

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # update the model so that model(train) gradually matches model(valid)
    history = model.fit(
        train_generator,
        shuffle=True,  # shuffle training_generator index set before each epoch
        steps_per_epoch=None,  # no shuffle if not None
        validation_data=valid_generator,
        validation_steps=step_size_valid,
        class_weight=label_weights_by_idx,
        callbacks=[tensorboard_callback],
        epochs=epochs)

    plot_model_fit_history(history)

    return model

    # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # acc_metric = keras.metrics.SparseCategoricalAccuracy()
    # train_writer = tf.summary.create_file_writer("logs/train/")
    # valid_writer = tf.summary.create_file_writer("logs/valid/")
    # train_step = test_step = 0

    # num_epochs = epochs
    # for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    #     train_step = test_step = 0
    #     train_writer = tf.summary.create_file_writer("logs/train/" + str(lr))
    #     valid_writer = tf.summary.create_file_writer("logs/valid/" + str(lr))
    #     model = create_model(target_size, dropout1, dropout2, labels)

    #     # see https://keras.io/api/optimizers/adam/
    #     optimizer = keras.optimizers.Adam(
    #         learning_rate=lr, # EXPERIMENTAL with [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #         beta_1=0.9, # The exponential decay rate for the 1st moment estimates.
    #         beta_2=0.999, # The exponential decay rate for the 2nd moment estimates
    #         epsilon=1e-07, # could be EXPERIMENTAL with [1e-7, 1e-1, 1e-0]
    #         amsgrad=False)

    #     for epoch in range(num_epochs):
    #         # Iterate through training set
    #         for batch_idx, (x, y) in enumerate(train_generator):
    #             with tf.GradientTape() as tape:
    #                 y_pred = model(x)
    #                 loss = loss_fn(y, y_pred)

    #             gradients = tape.gradient(loss, model.trainable_weights)
    #             optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    #             acc_metric.update_state(y, y_pred)

    #             with train_writer.as_default():
    #                 tf.summary.scalar("Loss", loss, step=train_step)
    #                 tf.summary.scalar(
    #                     "Accuracy", acc_metric.result(), step=train_step,
    #                 )
    #                 train_step += 1

    #         # Reset accuracy in between epochs (and for testing and test)
    #         acc_metric.reset_states()

    #         # Iterate through validator set
    #         for batch_idx, (x, y) in enumerate(valid_generator):
    #             y_pred = model(x)
    #             loss = loss_fn(y, y_pred)
    #             acc_metric.update_state(y, y_pred)

    #             with valid_writer.as_default():
    #                 tf.summary.scalar("Loss", loss, step=test_step)
    #                 tf.summary.scalar(
    #                     "Accuracy", acc_metric.result(), step=test_step,
    #                 )
    #                 test_step += 1

    #         acc_metric.reset_states()

    #     # Reset accuracy in between epochs (and for testing and validate)
    #     acc_metric.reset_states()


#------------ </fit_model> --------------#


def save_model(
    model,
    models_root_dir):
    model_dir_path = model_file_utils.save_model(models_root_dir, model)
    logger.info(f"saved model to model_dir_path: {model_dir_path}")
    loaded_model = model_file_utils.load_latest_model(models_root_dir)
    assert model_file_utils.models_are_equivalent(model, loaded_model)


def quick_evaluate_model(
    model,
    generator_name,
    generator):
    score = model.evaluate(generator)
    logger.info(f"{generator_name} loss: {score[0]}")
    logger.info(f"{generator_name} accuracy: {score[1]}")

#------------ <evaluatemodel> --------------#


def evaluate_model(
    model,
    test_generator,
    true_df,
    idx_to_label_map,
    labels):

    # use the model to get len(LABELS) prediction probabilities in [0..1) for each image
    Y_pred = model.predict(test_generator)

    # keep only the highest prediction value for each image
    y_pred = np.argmax(Y_pred, axis=1) 

    # convert y_pred idx values into y_pred labels and compare length of y_true
    pred_labels = [idx_to_label_map[i] for i in y_pred]
    y_true = true_df['label']
    true_labels = [idx_to_label_map[i] for i in y_true]
    assert len(pred_labels) == len(true_labels)

    # pull X_pred filesnames
    pred_filenames = test_generator.filenames
    true_filenames = true_df['filename']
    assert np.array_equal(pred_filenames, true_filenames)
    num_files = len(true_filenames)

    # Use the test_idx to plot image_files with true/pred labels  
    test_idx = generate_random_plot_idx(test_generator)
    true_v_pred_labels = [
        f"{true_labels[i]}/{pred_labels[i]}" for i in range(num_files)]
    plot_idxed_image_files_with_labels(
        None, true_filenames, true_v_pred_labels, test_idx)

    # compute and display the confusion matrix of true vs pred labels
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    logger.info('showing Confusion Matrix of true vs pred labels')
    logger.info(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    plt.show(block=False)

#------------ </evaluatemodel> --------------#


def get_imagefile_shape(csv_data_file, src_imgs_dir):
    with open(csv_data_file, "r") as f:
        line = f.readline().strip()
        imagefile = line.split(",")[0]
        imagepath = os.path.join(src_imgs_dir, imagefile)
        if os.path.isfile(imagepath):
            imagefile = Image.open(imagepath)
            return (imagefile.size[1], imagefile.size[0])

#------------ <run_pipeline> --------------#


def run_pipeline(params):
    '''use hyper-paramters to 
        create generators, 
        create model, 
        fit model,
        save model,
        evaluate_level and
        return evaluation results'''

    csv_data_file = params['csv_data_file']
    src_imgs_dir = params['src_imgs_dir']
    models_root_dir = params['models_root_dir']
    data_splits = params['data_splits']
    frame_subsample_rate = params['frame_subsample_rate']
    image_scale_factor = params['image_scale_factor']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    dropout1 = params['dropout1']
    dropout2 = params['dropout2']
    plot_random_images = params['plot_random_images']
    image_plots_only = params['image_plots_only']
    model = params['model']

    (imagefile_height, imagefile_width) = get_imagefile_shape(
        csv_data_file, src_imgs_dir)
    image_height = int(round(imagefile_height * image_scale_factor))
    image_width = int(round(imagefile_width * image_scale_factor))
    target_size = (image_height, image_width)

    labels = params['labels']
    label_to_idx_map = params['label_to_idx_map']
    idx_to_label_map = {
        label_to_idx_map[label]: label for label in label_to_idx_map.keys()}

    (generators, true_df, label_weights_by_idx) = create_generators(
        csv_data_file,
        src_imgs_dir,
        label_to_idx_map,
        idx_to_label_map,
        data_splits,
        frame_subsample_rate,
        batch_size,
        target_size,
        plot_random_images)

    (train_generator, valid_generator, test_generator) = generators

    # return early if image_plots_only is True
    if image_plots_only:
        return

    # train/fit the model only if model is None
    if model is None:

        model = create_model(
            target_size,
            dropout1,
            dropout2,
            labels)

        model = fit_model(
            model,
            train_generator,
            valid_generator,
            label_weights_by_idx,
            learning_rate,
            epochs)

        model_dir_path = save_model(
            model, 
            models_root_dir)

        history = evaluate_model(
            model,
            test_generator,
            true_df,
            idx_to_label_map,
            labels)

    return (history, model_dir_path)

#------------ </run_pipeline> --------------#

#------------ <tests> --------------#

def run_tests():
    logger.info("run_tests() not yet implementated")

#------------ </tests> --------------#


#------------ <main> --------------#

def main():

    usage="""
    usage:
        python cnn_image_classification.py ( help | run | test | latest | img-plots-only | <model_dir_path> )
    """

    # defaults to be overridden by command line argvs
    model=None
    image_plots_only=False
    model_dir_path=None

    # process command line argvs
    if len(sys.argv) > 1:
        argv1=sys.argv[1]
        if argv1 == 'help':
            print(usage)
            return 
        elif argv1 == 'run':
            pass
        elif argv1 == 'test':
            run_tests()
            return
        elif argv1 == 'latest':
            model=model_file_utils.load_latest_model()
            if model is None:
                logger.info("no latest model_dir_path found")
                logger.info("exiting now")
                return
        elif argv1 == 'img-plots-only':
            image_plots_only=True 
        else:
            model_dir_path=argv1
            model=model_file_utils.load_model(model_dir_path)
            if model is None:
                raise Exception(f"load_model({model_dir_path}) failed")

    # use command line args to modify parameters
    parameters={
        "csv_data_file": "../csv-data/S01E01-S01E02-data.csv",
        "src_imgs_dir": "../src-images/",
        "models_root_dir": "./models/",
        "labels": ['Junk', 'Common', 'Uncommon', 'Rare', 'Legendary'],
        "label_to_idx_map": {'Junk': 0, 'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Legendary': 4},
        "frame_subsample_rate": 24,
        "image_scale_factor": 0.5,
        "batch_size": 32,
        "dropout1": 0.25,
        "dropout2": 0.5,
        "epochs": 2,
        "data_splits": {'train_size': 0.70, 'valid_size': 0.20, 'test_size': 0.10},
        "learning_rate": 0.0001,
        "plot_random_images": False,
        "image_plots_only": image_plots_only,
        "model": model
    }

    # run the entire pipeline
    (history, model_dir_path) = run_pipeline(parameters)

    #------------ </main> --------------#


if __name__ == '__main__':
    with tf.device('/GPU'):
        main()
        print("done")
