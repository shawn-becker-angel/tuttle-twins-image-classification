
import datetime
import sys
import os
import re
import matplotlib.pyplot as plt
from matplotlib_utils import wait_for_click
import pandas as pd

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

def cleanup_filename(filename):
    pattern = r"[\!\#\$\%\^&*\(\)\[\]\{\}\\,.; \t\n\:]"
    return re.sub(pattern,'-', filename)

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
    wait_for_click()
    
    
#==============================================
# TESTS
#==============================================

def test_save_load_history():
    history = {'a':'b', 'c':'d', 'e':'f'}
    abs_path = save_history("/tmp/test_history", history)
    assert abs_path is not None
    restored_history = load_history(abs_path)
    assert restored_history is not None

#==============================================
# MAIN
#==============================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        history = load_history(json_file)
        plot_history(history)

    else:
        test_save_load_history()
        
    print("done")
