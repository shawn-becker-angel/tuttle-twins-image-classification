import os
import sys
import datetime
import numpy as np
from matplotlib_utils import plot_history
import type_utils
from dotenv import load_dotenv
load_dotenv()
HISTORY_ROOT_DIR = os.getenv('HISTORY_ROOT_DIR')

def save_history(history_root_dir, history):
    '''
    Save the given history (callback or dict) to a 
    unique history_path under history_root_dir and 
    return the history_path or return None if errors
    '''
    if history is not None:
        history_dict = type_utils.get_history_dict(history)
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
            max_ctime = ctime
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

def history_equal(history_A, history_B):
    '''Return True if equal, False if not equal, None if errors'''
    dict_A = type_utils.get_history_dict(history_A)
    keys_A = dict_A.keys()
    dict_B = type_utils.get_history_dict(history_B)
    keys_B =  dict_B.keys()
    if keys_A != keys_B:
        print(f"DEBUG: keys_A:{keys_A} != keys_B:{keys_B}")
        return False
    for key in  keys_A:
        val_A = dict_A[key]
        val_B = dict_B[key]
        if val_A != val_B:
            print(f"DEBUG: val_A:{val_A} != val_B:{val_B}")
            return False
    return True

def test_save_load():
    history = {
        "a":1,
        "b":2,
        "c":3
    }
    check_history_path = save_history(HISTORY_ROOT_DIR, history)
    check_history = load_history(check_history_path)
    os.remove(check_history_path)
    assert history_equal(history, check_history), "DEBUG: history check failure"

def tests():
    test_save_load()

def main(argv):
    usage = """
Usage:
  python history_file_utils.py (help | tests | latest | <history_path)
"""
    # defaults
    history_path = None

    argv1 = sys.argv[1] if len(argv) > 1 else 'help'
    if argv1 == 'help':
        print(usage)
        return
    elif argv1 in ['test', 'tests'] :
        tests()
        return
    elif argv1 == 'latest':
        history_path = find_latest_history_path(HISTORY_ROOT_DIR)
    else:
        history_path = argv1
    
    if history_path:
        print(f"loading history_path: {history_path}")
        history = load_history(history_path)
        if history is None:
            print("failed to load history_path:", history_path)
        else:
            plot_history(history)
        

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv)
    else:
        test_save_load()
    print("done history")
