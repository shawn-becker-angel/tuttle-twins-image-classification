# pip install matplotlib
# pip install opencv-python

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import random

from numpy.random import default_rng

def wait_for_click():
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    # plt.show(block=False)

def get_unique_random_ints(minVal: int, maxVal: int, N: int):
    R = maxVal-minVal
    if N > R:
        raise Exception("not possible to create N unique integers over range of {R} integers")
    rint = list(range(minVal,maxVal))
    np.random.shuffle(rint)
    return rint

def get_random_ints(minVal: int, maxVal: int, N: int):
    rints = list(range(N))
    np.random.shuffle(rints)
    buckets = list(range(minVal,maxVal))
    vals = [rints[i]/bucket[i]]
    return rint

def generate_random_plot_idx(generator):
    N = generator.n
    if N < 1:
        msg = "ERROR: generator.n is not > zero"
        raise Exception(msg)
    
    rints = get_unique_random_ints(0, N, N)
    if len(rints) != N:
        msg = f"ERROR: rints:{len(rints)} != N:{N}"
        raise Exception(msg)
    
    return rints

def plot_idxed_image_files_with_labels(name, image_files, labels, plot_idx):
    assert len(image_files) == len(labels) == len(plot_idx), "ERROR: length failures"
    N = min(len(image_files),12) # number of images
    fig = plt.figure(figsize=(12,8))
    for n in range(N):
        i = plot_idx[n]
        plt.subplot(3,4,n+1)
        image = cv2.imread(image_files[i])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.imshow(image)   
        plt.axis('off')
        title = f"{name}[{i}]: {labels[i]}" if name is not None else labels[i]
        plt.title(title)
    name_cnt =  f"{N} {name}" if name is not None else f"{N}"
    print(f"Showing {name_cnt} images and labels")
    wait_for_click()

def plot_idxed_generator_images(name, generator, plot_idx, idx_to_label_map=None):
    assert len(plot_idx) > 0, "ERROR: empty plot_idx"
    if idx_to_label_map is not None:
        X, y = generator.next()
    else:
        X = generator.next()
        y = None

    fig = plt.figure(figsize=(12,8))
    
    # trim plot_idx to not exceed the new N
    N = min(len(X),12) # number of images
    plot_idx = [plot_idx[n] for n in plot_idx if plot_idx[n] < N] 
    
    for n in range(N):
        i = plot_idx[n]
        plt.subplot(3,4,n+1)
        plt.imshow(X[i,:,:,:])   
        plt.axis('off')
        label = "unknown" if y is None else idx_to_label_map[y[i]]
        plt.title(f"{name}[{i}]: {label}")

    legend = "images only" if y is None else "images and labels"
    print(f"Showing {N} {name} {legend}")
        
    wait_for_click()

def plot_histogram(title: str='Title', data: List[float]=[], with_normal: bool=True):
    import matplotlib.pyplot as plt

    plt.hist(data, 50, density=True)
    plt.ylabel('Probability')
    plt.xlabel('Data');

    if with_normal:
        from scipy.stats import norm
        mu, std = norm.fit(data) 
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = f"{title} mu:{mu:.2f} std:{std:.2f}"
    
    plt.title(title)
    plt.show()
    wait_for_click()

def plot_model_fit_history(history):
    # see https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    
    # list all metrics in history
    print(history.history.keys())
    
    num_epochs = len(history.history['Accuracy'])
    if num_epochs < 2:
        print("skipping plot_model_fit_history() since num_epochs is less than 2")
        return
    
    # plot accuracy metrics per epoch (each metric must be in history.history.keys)
    plt.plot(history.history['Accuracy'])
    plt.plot(history.history['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # plot loss metrics per epoch (each metric must be in history.history.keys)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    wait_for_click()


#==============================================
# TESTS
#==============================================

def test_plot_gamma_histogram():
    shape = 1.5
    N = 100_000
    s = np.random.standard_gamma(shape, N)
    assert len(s) == N
    plot_histogram(title="gamma distribution", data=s )

def test_plot_rand_int_histogram():
    s = []
    minVal = 0
    maxVal = 100
    N = 100_000
    for i in range(N):
        s.append(np.random.randint(minVal,maxVal))
    assert len(s) == N
    assert np.min(s) >= minVal
    maxS = np.max(s)
    if not maxS < maxVal:
        print(f"ERROR: maxS:{maxS} not < maxVal:{maxVal}")
    plot_histogram(title="randint distribution", data=s )

#==============================================
# MAIN
#==============================================

if __name__ == "__main__":
    test_plot_gamma_histogram()
    test_plot_rand_int_histogram()
    print("done")