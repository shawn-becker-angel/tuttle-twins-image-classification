# pip install matplotlib
# pip install opencv-python

import os
import sys
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import default_rng
from typing import List
import random
import type_utils
from sklearn.metrics import ConfusionMatrixDisplay
from dotenv import load_dotenv

load_dotenv()
HISTORY_ROOT_DIR = os.getenv('HISTORY_ROOT_DIR')
CM_DICT_ROOT_DIR = os.getenv("CM_DICT_ROOT_DIR")

assert HISTORY_ROOT_DIR is not None, "HISTORY_ROOT_DIR not found in dotenv"
assert CM_DICT_ROOT_DIR is not None, "CM_DICT_ROOT_DIR not found in dotenv"

def wait_for_click(timeout_seconds=-1) -> int:
    '''Wait for user input and return True if a key was pressed, 
    False if a mouse button was pressed and None if no input was 
    given within timeout seconds. Negative values deactivate timeout.'''

    print(f"Click key or mouse in window to close.")
    result = plt.waitforbuttonpress(timeout_seconds)
    plt.close("all")
    # plt.show(block=False)
    return result

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
    fig_title = name
    fig = plt.figure(figsize=(12,8), num=fig_title)
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
    fig_title = name
    fig = plt.figure(figsize=(12,8),num=fig_title)
    
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

def plot_history(history, timeout=-1, default_save_figure_dir=HISTORY_ROOT_DIR, save_figure_path=None):
    # see https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    
    history_dict = type_utils.get_history_dict(history)

    # list all metrics in history_dict
    if not save_figure_path:
        print(history_dict.keys())
    
    num_epochs = len(history_dict['Accuracy'])
    if num_epochs < 2:
        print("skipping plot_history() since num_epochs is less than 2")
        return
    
    fig_title = "my-history"
    fig = plt.figure(figsize=(12,6), num=fig_title)

    # plot accuracy metrics per epoch (each metric must be in history_dict.keys)
    plt.subplot(1,2,1)
    plt.plot(history_dict['Accuracy'])
    plt.plot(history_dict['val_Accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    if not save_figure_path:
        print("Showing accuracy metrics per epoch")
    
    # plot loss metrics per epoch (each metric must be in history_dict.keys)
    plt.subplot(1,2,2)
    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    if not save_figure_path:
        print("Showing loss metrics per epoch")

    if save_figure_path is not None:
        plt.savefig(save_figure_path)
        print("Saved ", save_figure_path)
        # return without showing
        return save_figure_path

    if default_save_figure_dir is not None:
        import matplotlib as mpl
        mpl.rcParams["savefig.directory"] = default_save_figure_dir

    plt.show()

def plot_cm_dict(cm_dict, timeout=-1, default_save_figure_dir=CM_DICT_ROOT_DIR, save_figure_path=None):
    assert type_utils.is_cm_dict(cm_dict)
    cm = cm_dict['cm']
    labels = cm_dict['labels']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)

    if save_figure_path is not None:
        plt.savefig(save_figure_path)
        print("Saved confusion matrix of true vs pred labels to ", save_figure_path)
        # return without showing
        return save_figure_path

    if default_save_figure_dir is not None:
        import matplotlib as mpl
        mpl.rcParams["savefig.directory"] = default_save_figure_dir

    print("Showing confusion matrix of true vs pred labels")
    wait_for_click(timeout)

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

def run_tests():
    test_plot_gamma_histogram()
    test_plot_rand_int_histogram()

#==============================================
# MAIN
#==============================================

def main(argv):
    usage="""
    usage:
        python matplotlib_utils.py ( help | test )
    """
    argv1 = argv[1] if len(argv) > 0 else 'help'

    # defaults to be overridden by command line argvs

    # process command line argvs
    if argv1 == 'help':
        print(usage)
        return 
    elif argv1 in ['test', 'tests']:
        run_tests()
        return 


if __name__ == "__main__":
    main(sys.argv)
