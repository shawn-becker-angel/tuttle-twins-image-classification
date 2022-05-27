import tensorflow as tf   # version 2.1.0
import pandas as pd

IMAGE_SIZE = 128
IMAGE_HEIGHT = IMAGE_SIZE
IMAGE_WIDTH = IMAGE_SIZE

# batch_size must evenly divide the length of the dataset exactly
# because for the test set, you should sample the images exactly once
BATCH_SIZE = 32

train_df = pd.read_csv("../csv-data/train_data.csv",header=None, names=['filename','label'])
test_df = pd.read_csv("../csv-data/test_data.csv",header=None, names=['filename','label'])

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

def Gen():
  gen = img_gen.flow_from_dataframe(
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
  
  for (x,y) in gen:
    yield (x,y)

NUM_CLASSES = 5
IMAGES_SHAPE = [BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,3]
LABELS_SHAPE = [BATCH_SIZE,NUM_CLASSES]
ds = tf.data.Dataset.from_generator(  
  generator=Gen,  
  output_types=(tf.float32, tf.float32),  
  output_shapes=(IMAGES_SHAPE, LABELS_SHAPE)
)

it = iter(ds)
batch = next(it)
print(batch)
