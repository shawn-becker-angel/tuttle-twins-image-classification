import tensorflow as tf   # version 2.1.0

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
flowers_root_path = tf.keras.utils.get_file(origin=DATA_URL, fname='flower_photos', untar=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
def Gen():
  gen = img_gen.flow_from_directory(flowers_root_path)
  for (x,y) in gen:
    yield (x,y)

ds = tf.data.Dataset.from_generator(  
  Gen,  output_types=(tf.float32, tf.float32),  output_shapes=([32,256,256,3], [32,5]))

it = iter(ds)
batch = next(it)
print(batch)
