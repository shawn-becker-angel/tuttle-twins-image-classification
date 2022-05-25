# from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial18-customdata-images/2_csv_file.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# https://stackoverflow.com/a/55938423
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# pip install scikit-learn
from sklearn.model_selection import train_test_split

# pip install opencv-python
import cv2

image_source_directory = "../src-images/"
if not os.path.isdir(image_source_directory):
    raise Exception(f"no such directory: {image_source_directory}")

# read the train data file into train_df
train_csv_file =  "../csv-data/train_data.csv"
if not os.path.isfile(train_csv_file):
    raise Exception(f"no such file: {train_csv_file}")
train_df = pd.read_csv(train_csv_file,header=None, names=['file_name','label'])
print("train_df.shape:", train_df.shape)

# read the test data file into test_df
test_csv_file =  "../csv-data/test_data.csv"
if not os.path.isfile(test_csv_file):
    raise Exception(f"no such file: {test_csv_file}")
test_df = pd.read_csv(test_csv_file,header=None, names=['file_name','label'])
print("test_df.shape:", test_df.shape)

# combine train_df and test_df into a single df
df = pd.concat([train_df,test_df],axis=0)
print("df.shape:", df.shape)

# extract file_names and labels
file_names = df["file_name"].tolist()
labels = df["label"].tolist()

N = len(file_names)
assert len(labels) == N

# convert each label into label_ones tensor
# see https://www.datacamp.com/tutorial/cnn-tensorflow-python

# create the str label to int label converter 
# verify that label_converter covers all label values
label_converter = {'Junk':0, 'Common':1, 'Uncommon':2, 'Rare':3, 'Legendary':4 }
K = label_conv_count = len(label_converter)
label_value_count = len(set(labels))
# it is possible that the labels list does not 
# contain all possible label values defined in the label_converter
assert label_value_count <= label_conv_count 

# check spelling of label_converter keys
if label_value_count == label_conv_count:
    assert set(label_converter.keys()) == set(labels)

# use the label_converter to convert str labels into int_labels
int_labels = []
int_labels.append([label_converter[item] for item in labels])
int_labels = np.array(int_labels)

# categorize int labels to use K columns of flags
# e.g. 0:[1,0,0,0,0], 1:[[0,1,0,0,0]], ... , 4:[[0,0,0,0,1]]
from keras.utils import to_categorical
label_ones = to_categorical(int_labels)
label_ones = label_ones.reshape(-1,K)
# shape should be (N,K)
print("label_ones.shape:", label_ones.shape)

# read_and_resize_imgs 
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

size = 128
dim = [size, size]
imgs = []
for i in range(N):
    jpg_file = os.path.join(image_source_directory, file_names[i])
    assert os.path.isfile(jpg_file)
    img = cv2.imread(jpg_file)
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dim, method= preserve_aspect_ratio=False, antialias=True)
    imgs.append(img)

imgs = np.array(imgs)
print("imgs.shape:", imgs.shape)
# (N, size, size, 3) 

assert len(imgs) == len(label_ones)

# display_2_imgs
plt.figure(figsize=[5,5])

# Display a random image
def display_random_image(subplot):
    idx = random.randrange(0,N)
    plt.subplot(subplot)
    # curr_img = np.reshape(imgs[idx], size, size, 3)
    curr_img = imgs[idx]
    curr_lbl = labels[idx]
    plt.imshow(curr_img)
    plt.title(f"idx:{idx} ({curr_lbl})")

display_random_image(121)
display_random_image(122)

# normalize imgs
imgs = imgs / np.max(imgs)
print("min:", np.min(imgs), "max:", np.max(imgs))

# split the imgs and label_ones into train and test data
X_train, X_test, y_train, y_test = train_test_split(imgs, label_ones, test_size=0.2, random_state=42)

train_X = X_train.reshape(-1, size, size, 3)
test_X = X_test.reshape(-1, size, size, 3)
train_y = y_train
test_y = y_test

# Data Preprocessing
# The images are of size x size for a size x size vector V

print("X_train shape:", X_train.shape, "min:", np.min(X_train), "max:", np.max(X_train))
print("X_test shape:", X_test.shape, "min:", np.min(X_test), "max:", np.max(X_test))

# The Deep Neural Network

# You'll use three convolutional layers:
#
# The first layer will have 32-3 x 3 filters,
# The second layer will have 64-3 x 3 filters and
# The third layer will have 128-3 x 3 filters.
#
# In addition, there are three max-pooling layers, each of the size 2 x 2.
#
# Input 128x128x3
# Convolution 32x 3x3 filter
# Max Pooling 2x2
# Convolution 64x 3x3 filter
# Max Pooling 2x2
# Convolution 128x 3x3 filter
# Max Pooling 2x2
# Flatten
# Dense Layer 128 units
# Output Layer 5 units

# Network Parameters

EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
N_INPUT = size
N_CLASSES = K

# create the placeholders
x = tf.compat.v1.placeholder("float", [None, N_INPUT, N_INPUT, 1])
y = tf.compat.v1.placeholder("float", [None, N_CLASSES])

# Initialize the variables - not needed for tensorflow 2.0.x.x
# init = tf.global_variables_initializer()

# Create wrappers
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# define your weights and biases variables
# https://stackoverflow.com/a/59906649
# https://stackoverflow.com/a/69264194
# https://stackoverflow.com/a/64258851
weights = {
    'wc1': tf.Variable('W0', shape=(3,3,1,32), initializer=tf.keras.initializers.glorot_normal),
    'wc2': tf.Variable('W1', shape=(3,3,32,64), initializer=tf.keras.initializers.glorot_normal),
    'wc3': tf.Variable('W2', shape=(3,3,64,128), initializer=tf.keras.initializers.glorot_normal),
    'wd1': tf.Variable('W3', shape=(4*4*128,128), initializer=tf.keras.initializers.glorot_normal),
    'out': tf.Variable('W6', shape=(128,N_CLASSES), initializer=tf.keras.initializers.glorot_normal),
}
biases = {
    'bc1': tf.Variable('B0', shape=(32), initializer=tf.keras.initializers.glorot_normal),
    'bc2': tf.Variable('B1', shape=(64), initializer=tf.keras.initializers.glorot_normal),
    'bc3': tf.Variable('B2', shape=(128), initializer=tf.keras.initializers.glorot_normal),
    'bd1': tf.Variable('B3', shape=(128), initializer=tf.keras.initializers.glorot_normal),
    'out': tf.Variable('B4', shape=(N_CLASSES), initializer=tf.keras.initializers.glorot_normal),
}

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate Model Node

# Initializing the variables
init = tf.global_variables_initializer()

# check whether the index of the maximum value of the predicted image is equal to 
# the actual labeled image. And both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(EPOCHS):
        for batch in range(len(train_X)//BATCH_SIZE):
            batch_x = train_X[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(train_X))]
            batch_y = train_y[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        
        print("Epoch " + str(i) + \
            ", Loss= " + "{:.6f}".format(loss) + \
            ", Training Accuracy= " + "{:.5f}".format(acc))

        # print("Optimization Finished!")

        # Calculate accuracy against all test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

# put your model evaluation into perspective and plot the 
# accuracy and loss plots between training and validation data

plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

