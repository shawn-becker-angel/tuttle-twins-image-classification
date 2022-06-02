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
        plt.imshow(image)   
        plt.axis('off')
        plt.title(f"{name}[{i}]: {labels[i]}")
    print(f"Showing {N} {name} images and labels")
    wait_for_click()

def plot_idxed_generator_images(name, generator, plot_idx, idx_to_label_map=None):
    assert len(plot_idx) > 0, "ERROR: empty plot_idx"
    generator.reset()
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