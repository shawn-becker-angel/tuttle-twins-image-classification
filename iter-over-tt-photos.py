import tensorflow as tf   # version 2.1.0
import pandas as pd
from tensorflow import keras

# pip uninstall keras-preprocessing
# Install the keras-preprocessing module from the following git link:
# pip install git+https://github.com/keras-team/keras-preprocessing.git

from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 128
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE
BATCH_SIZE = 32

train_df = pd.read_csv("../csv-data/train_data.csv",header=None, names=['filename','label'])
test_df = pd.read_csv("../csv-data/test_data.csv",header=None, names=['filename','label'])

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

x_train = train_generator.filenames 
y_train = train_generator.labels

EPOCHS = 20
for e in range(EPOCHS):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=BATCH_SIZE):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break



N = len(train_generator)
print(f'Total number of batches - {N}')
for n, i in enumerate(train_generator):
    batch_data = i[0]
    print(n, batch_data[0].shape)

# TRY to access element out of bound to see if there really exists more than 30 elements.
print(f'{train_generator[32]}')

