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
    '''Return a list of N unique int values x such that minVal <= x < maxVal'''
    rints = []
    for n in range(N*2):
        rint = np.random.randint(minVal, maxVal)
        if not rint in rints:
            rints.append(rint)
        if len(rints) == N:
            return rints

def plot_random_generator_images_with_labels(name, generator, idx_to_label_map):
    generator.reset()
    X, y = generator.next()
    fig = plt.figure(figsize=(12,8))
    N = min(len(X),12) # number of images
    random_ints = get_unique_random_ints(0,len(X),N)
    for n in range(N):
        i = random_ints[n] # random i'th image in X
        plt.subplot(3,4,n+1)
        plt.imshow(X[i,:,:,:])   
        plt.axis('off')
        index = y[i]
        plt.title(f"{name}[{i}]: {idx_to_label_map[index]}")
    print(f"Showing {N} {name} images and labels")
    wait_for_click()

def plot_random_generator_images_no_labels(name, generator):
    X = next(generator)
    X = X[1:]  # strip off batch dimension
    N = min(len(X),12) # number of images
    fig = plt.figure(figsize=(12,8))
    random_ints = get_unique_random_ints(0,len(X),N)
    for n in range(N):
        i = random_ints[n]
        plt.subplot(3,4,n+1)
        plt.imshow(X[i])
        plt.title(f"{name}[{i}]: unknown")
    print(f"Showing {name} images")
    wait_for_click()

def plot_random_imagefiles_with_labels(name, image_files, labels):
    N = min(len(X),12) # number of images
    fig = plt.figure(figsize=(12,8))
    random_ints = get_unique_random_ints(0,len(image_files)+1,N)
    for n in range(N):
        i = random_ints[n]
        plt.subplot(3,4,n+1)
        image = cv2.imread(image_files[i])
        plt.imshow(image)   
        plt.axis('off')
        plt.title(f"{name}[{i}]: {labels[i]}")
    print(f"Showing {name} images and labels")
    wait_for_click()


def plot_confusion_matrix(labels, y_pred, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    print('Confusion Matrix')
    # confusion_matrix(validation_generator.classes, y_pred)
    print('Classification Report')
    target_names = labels
    # classification_report(validation_generator.classes, y_pred, target_names=target_names)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
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