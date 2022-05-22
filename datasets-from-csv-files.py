# from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial18-customdata-images/2_csv_file.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

HOME_DIR = os.path.expanduser('~') 

train_csv_file =  "train_data.csv"
if not os.path.isfile(train_csv_file):
    raise Exception(f"no such file: {train_csv_file}")

directory = "source_images/"
if not os.path.isdir(directory):
    raise Exception(f"no such directory: {directory}")


df = pd.read_csv(train_csv_file,header=None, names=['file_name','label'])

file_paths = df["file_name"].values
labels = df["label"].values
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    return image, label

def augment(image, label):
    # data augmentation here
    print(f"augment image:{image} label:{label}")
    return image, label


ds_train = ds_train.map(read_image).map(augment).batch(2)

for epoch in range(10):
    for x, y in ds_train:
        # train here
        pass

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)

#==================================
# TESTS
#==================================

def test_read_image():
    image_file =  file_paths.iloc[0]
    label = labels.iloc[0]
    img, img_class = read_image(image_file, label)
    print(img)
    print(img_class)

if __name__ == '__main__':
    test_read_image()
