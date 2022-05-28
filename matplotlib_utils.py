# pip install matplotlib
import matplotlib
print("matplotlib.version:", matplotlib.__version__)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import sys
import os
import re

def get_unique_random_ints(minVal: int, maxVal: int, N: int):
    '''Return a list of N unique int values x such that minVal <= x < maxVal'''
    s = set()
    for n, x in enumerate(list(np.random.randint(minVal,maxVal,N*2))) : 
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
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    plt.show(block=False)
    print("closed")

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
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.show(block=False)
    plt.close("all")
    print("closed")

def plot_random_imagefiles_with_labels(name, image_files, labels):
    N = 12 # number of images
    fig = plt.figure(figsize=(12,8))
    random_ints = get_unique_random_ints(0,len(X)+1,N)
    for n in range(N):
        i = random_ints[n]
        plt.subplot(3,4,n+1)
        image = imread(image_files[i])
        plt.imshow(image)   
        plt.axis('off')
        plt.title(f"{name}[{i}]: {labels[i]}")
    print(f"Showing {name} images and labels")
    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.show(block=False)
    plt.close("all")
    print("closed")

def cleanup_filename(filename):
    pattern = r"[\!\#\$\%\^&*\(\)\[\]\{\}\\,.; \t\n\:]"
    return re.sub(pattern,'-', filename)

def save_history(filename, history):
    '''
    Saves history object to a json_file
    named <filename>-<datetime>.json
    Return None if history could not be 
    saved, otherwise return the path
    of the saved json file.
    '''
    if not isinstance(history, dict):
        print(f"history: {type(history)} {str(history)}")
        print( dir(history.__class__) )
        return None

    df = None
    try:
        df = pd.DataFrame.from_dict(history)
    except ValueError as ve:
        for key in history.keys():
            val = history[key]
            history[key] = list(val)
        df = pd.DataFrame.from_dict(history)
    except Exception as exp:
        print(f"{type(exp)} {str(exp)}")
        raise

    if df is None:
        print(f"ERROR: unable to convert history to dict and json file")
        return None

    # use current utc time in milliseconds as a random name
    dt = datetime.datetime.utcnow().isoformat()

    filename = cleanup_filename(filename)
    json_file = f'{filename}-{dt}.json'
    json_file_path = os.path.abspath(json_file)

    df.to_json(json_file_path)
    return json_file_path

def load_history(json_file_path: "PathLike[str]"):
    '''Returns the loaded history or None if load failed'''
    try:
        df = pd.read_json(json_file_path)
        history = df.to_dict()
        return history
    except Exception as exp:
        print("load_history failed")
    return None

def plot_history(name: str, history) -> None:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = len(acc)
    assert len(val_acc) == epochs

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    assert len(loss) == epochs
    assert len(val_loss) == epochs

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    print(f"Click key or mouse in window to close.")
    plt.waitforbuttonpress()
    plt.close("all")
    plt.show(block=False)
    print("closed")

def plot_confusion_matrix(labels, y_pred, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = labels
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()


#==============================================
# TESTS
#==============================================

def test_save_load_history():
    history = {'a':'b', 'c':'d', 'e':'f'}
    abs_path = save_history("/tmp/test_history", history)
    assert abs_path is not None
    restored_history = load_history(abs_path)
    # assert restored_history == history

def test_get_unique_random_ints():
    minVal = 0
    maxVal = 100
    N = 10
    x_list = get_unique_random_ints(minVal, maxVal, N)
    assert len(x_list) == N
    assert min(x_list) >= minVal
    assert max(x_list) < maxVal
    check = [x_list[i] for i in range(N)]
    print(check)


    maxVal = 5
    try:
        x_list = get_unique_random_ints(minVal, maxVal, N)
        check = [x_list[i] for i in range(N)]
        print(check)
    except Exception as exp:
        assert "ERROR" in str(exp)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        history = load_history(json_file)
        plot_history(history)

    else:
        test_save_load_history()
        test_get_unique_random_ints()
