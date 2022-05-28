# pip install matplotlib
# pip install opencv-python

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def wait_for_click():
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    # plt.show(block=False)

def get_unique_random_ints(minVal: int, maxVal: int, N: int):
    '''Return a list of N unique int values x such that minVal <= x < maxVal'''
    s = set()
    for n, x in enumerate(list(np.random.uniform(minVal,maxVal,N*2))) : 
        if x not in s:
            s.add(x)
    if len(s) >= N:
        return list(s)[:N]
    raise Exception(f"ERROR: can't find {N} unique ints in {N*2} attempts")

def plot_random_generator_images_with_labels(name, generator):
    X, y = next(generator)
    decode_index = {b: a for a, b in generator.class_indices.items()}
    fig = plt.figure(figsize=(12,8))
    N = 12 # number of images
    random_ints = get_unique_random_ints(0,len(X),N)
    for n in range(N):
        i = random_ints[n]
        plt.subplot(3,4,n+1)
        plt.imshow(X[i])   
        plt.axis('off')
        index = np.argmax(y[i])
        plt.title(f"{name}[{i}]: {decode_index[index]}")
    print(f"Showing {name} images and labels")
    wait_for_click()

def plot_random_generator_images_no_labels(name, generator):
    X = next(generator)
    X = X[1:]  # strip off batch dimension
    N = 12 # number of images
    fig = plt.figure(figsize=(12,8))
    random_ints = get_unique_random_ints(0,len(X)+1,N)
    for n in range(N):
        i = random_ints[n]
        plt.subplot(3,4,n+1)
        plt.imshow(X[i])
        plt.title(f"{name}[{i}]: unknown")
    print(f"Showing {name} images")
    wait_for_click()

def plot_random_imagefiles_with_labels(name, image_files, labels):
    N = 12 # number of images
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

def plot_histogram(title: str='Title', data: List[float]=[], shape: float=1, scale: float=1, with_normal: bool=True):
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

def test_N_lists_of_unique_random_ints():
    data = list()
    for i in range(100):
        check = test_get_list_of_unique_random_ints()
        data.extend(check)
    plot_histogram(title="uniform random ints", data=data, shape=1., scale=1.)
        
def test_get_list_of_unique_random_ints():
    minVal = 0
    maxVal = 100
    N = 10
    x_list = get_unique_random_ints(minVal, maxVal, N)
    assert len(x_list) == N
    assert min(x_list) >= minVal
    assert max(x_list) < maxVal
    check = [x_list[i] for i in range(N)]
    # print(check)
    return check

def test_plot_gamma_histogram():
    shape, scale = 2., 1. # mean and width
    s = np.random.standard_gamma(shape, 1000000)
    plot_histogram(title="gamma distribution", data=s, shape=2., scale=1. )


#==============================================
# MAIN
#==============================================

if __name__ == "__main__":
    test_N_lists_of_unique_random_ints()
    test_plot_gamma_histogram()
    print("done")