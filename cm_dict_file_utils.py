import os
import sys
import numpy as np
import json
from typing import List
import datetime
from matplotlib_utils import plot_cm_dict
from numpy_array_encoder import NumpyArrayEncoder
from type_utils import is_cm_dict, make_cm_dict
from dotenv import load_dotenv
load_dotenv()
CM_DICT_ROOT_DIR = os.getenv('CM_DICT_ROOT_DIR')

def save_cm_dict(cm_dict_root_dir: str, cm_dict: dict) -> str:
    '''Save a cm_dict to a unique cm_dict_path and return the cm_dict_path'''
    assert  cm_dict_root_dir is not None, "DEBUG: undefined cm_dict_root_dir"
    assert is_cm_dict(cm_dict), "DEBUG: cm_dict is not cm_dict"
    dt = datetime.datetime.utcnow().isoformat()
    cm_dict_path = os.path.join(cm_dict_root_dir, f"cm-{dt}.json")
    with open(cm_dict_path, "w") as write_file:
        json.dump(cm_dict, write_file, cls=NumpyArrayEncoder)
    return cm_dict_path

def load_cm_dict(cm_dict_path: str) -> dict:
    '''Load and return a cm_dict from the given cm_dict_path'''
    with open(cm_dict_path, "r") as read_file:
        raw_cm_dict = json.load(read_file)
        cm = np.asarray(raw_cm_dict["cm"])
        labels = raw_cm_dict['labels']
        cm_dict = make_cm_dict(cm, labels)
        return cm_dict

def find_latest_cm_dict_path(cm_dict_root_dir: str) -> str:
    assert  cm_dict_root_dir is not None, "DEBUG: undefined cm_dict_root_dir"
    cm_dict_paths = [os.path.join(cm_dict_root_dir,f) for f in os.listdir(cm_dict_root_dir) if f.startswith('cm-')]
    if len(cm_dict_paths) == 0:
        print(f"no cm_dict_paths found under {cm_dict_root_dir}")
        return None
    latest_cm_dict_path = None
    max_ctime = 0.0
    for cm_dict_path in cm_dict_paths:
        ctime = os.path.getctime(cm_dict_path)
        if ctime > max_ctime:
            latest_cm_dict_path = cm_dict_path
            max_ctime = ctime
    if latest_cm_dict_path is not None:
        return latest_cm_dict_path
    return None

def load_latest_cm_dict(cm_dict_root_dir) -> dict:
    '''
    Find, load, and return the latest cm_dict
    saved under cm_dict_root_dir or return None if errors
    '''
    assert  cm_dict_root_dir is not None, "DEBUG: undefined cm_dict_root_dir"
    cm_dict_path = find_latest_cm_dict_path(cm_dict_root_dir)
    if cm_dict_path is not None:
        return load_cm_dict(cm_dict_path)
    return None

def cm_dict_equal(cm_dict_A, cm_dict_B):
    '''Return True if equal, False if not equal, None if errors'''
    if not is_cm_dict(cm_dict_A):
        print("DEBUG: cm_dict_A is not cm_dict")
        return False
    if not is_cm_dict(cm_dict_B):
        print("DEBUG: cm_dict_B is not cm_dict")
    cmA = cm_dict_A['cm']
    cmB = cm_dict_B['cm']
    if cmA.shape != cmB.shape:
        print("DEBUG: cmA.shape:{cmA.shape} != cmB.shape:{cmB.shape}")
        return False
    lstA = list(cmA.flatten())
    lstB = list(cmB.flatten())
    if lstA != lstB:
        print(f"DEBUG: lstA:{lstA} != lstB:{lstB}")
        return False
    labA = cm_dict_A['labels']
    labB = cm_dict_B['labels']
    if labA != labB:
        print(f"DEBUG: labA:{labA} != labB:{labB}")
        return False
    return True


def test_save_load():
    def check_save_load(cm_array, labels):
        try:
            assert isinstance(cm_array, np.ndarray), "DEBUG: cm_array is not np.ndarray"
            assert isinstance(labels, list), "DEBUG: labels is not list"
            (rows, cols) = cm_array.shape
            assert rows == cols, "DEBUG: cm_array rows != cols"
            assert isinstance(labels, list), "DEBUG: labels is not list"
            assert len(labels) == rows, "DEBUG: len(labels) != rows"
            assert isinstance(labels[0], str), "DEBUG: labels[0] is not str"
            cm_dict = make_cm_dict(cm_array, labels)
            assert is_cm_dict(cm_dict), "DEBUG: cm_dict is not cm_dict"
            cm_check_path = save_cm_dict(CM_DICT_ROOT_DIR, cm_dict)
            cm_check = load_cm_dict(cm_check_path)
            assert is_cm_dict(cm_check), "DEBUG: cm_check is not cm_dict"
            os.remove(cm_check_path)
            assert cm_dict_equal(cm_check, cm_dict), "DEBUG: cm_check not equals cm_dict"
            return True
        except AssertionError as err:
            print(f"{type(err)} {str(err)}")
            print("check_save_load returning False")
            return False

    labels = ["ab","cd","ef"]
    assert check_save_load(np.array([[1,2,3],[4,5,6],[7,8,9]]), labels), "DEBUG: int check failed"
    assert check_save_load(np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]), labels), "DEBUG: flt check failed"
    assert check_save_load(np.array([["aa","bb","cc"],["dd","ee","ff"],["gg","hh","ii"]]), labels), f"DEBUG: str check failed"
    return True

def tests():
    test_save_load()

def main(argv):
    usage = """
Usage:
  python cm_dict_file_utils.py (help | test | latest | <cm_dict_path)
"""
    # defaults
    cm_dict_path = None

    argv1 = argv[1] if len(argv) > 1 else 'help'
    if argv1 == 'help':
        print(usage)
        return
    elif argv1 in ['test', 'tests']:
        tests()
        return
    elif argv1 == 'latest':
        cm_dict_path = find_latest_cm_dict_path(CM_DICT_ROOT_DIR)
    else:
        cm_dict_path = argv1
    
    if cm_dict_path is not None:
        print(f"loading cm_dict_path:", cm_dict_path)
        cm_dict = load_cm_dict(cm_dict_path)
        if cm_dict is None:
            print("failed to load cm_dict_path:", cm_dict_path)
        else:
            plot_cm_dict(cm_dict)


if __name__ == '__main__':
    main(sys.argv)
    print("done cm-dict")
