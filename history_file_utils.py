import os
import datetime
import numpy as np
import keras

def save_history(history_root_dir, history):
    '''
    Save the given history (callback or dict) to a 
    unique history_path under history_root_dir and 
    return the history_path or return None if errors
    '''
    if history is not None:
        if isinstance(history, keras.callbacks.History):
            history_dict = history.history
        elif isinstance(history, dict):
            history_dict = history
        if history_root_dir is not None:
            dt = datetime.datetime.utcnow().isoformat()
            history_path = os.path.join(history_root_dir, f"history-{dt}.npy")
            np.save(history_path, history_dict, allow_pickle=True)
            print("INFO:history written to ", history_path)
            return history_path
    return None

def find_latest_history_path(history_root_dir):
    '''
    Return the path of the latest history_path under history_root_dir
    or return None if errors
    '''
    history_paths = [os.path.join(history_root_dir,f) for f in os.listdir(history_root_dir) if f.startswith('history-')]
    if len(history_paths) == 0:
        print(f"no history_paths found under {history_root_dir}")
        return None
    latest_history_path = None
    max_ctime = 0.0
    for history_path in history_paths:
        ctime = os.path.getctime(history_path)
        if ctime > max_ctime:
            latest_history_path = history_path
    if latest_history_path is not None:
        return latest_history_path
    return None

def load_history(history_path):
    '''
    Load and return a history_dict from the given history_path
    or return None if errors
    '''
    if history_path is not None and os.path.isfile(history_path):
        history_nparray = np.load(history_path,allow_pickle=True)
        if history_nparray is not None and history_nparray.size >= 1:
            history_dict = history_nparray.item(0)
            if isinstance(history_dict,dict):
                return history_dict
    return None

def load_latest_history(history_root_dir):
    '''
    Find, load, and return the latest history_dict
    saved under history_root_dir or return None if errors
    '''
    history_path = find_latest_history_path(history_root_dir)
    if history_path is not None:
        return load_history(history_path)
    return None
