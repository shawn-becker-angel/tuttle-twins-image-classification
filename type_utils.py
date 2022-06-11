import keras
import numpy as np
from typing import List

def get_history_dict(history):
    if isinstance(history, keras.callbacks.History):
        history_dict = history.history
    elif isinstance(history, dict):
        history_dict = history
    assert isinstance(history_dict, dict), "history_dict is not a dict"
    return history_dict


def is_cm_dict(cm_dict) -> bool:
    '''Return True if the given dict contains all
    of the keys and values required of a cm_dict'''
    if not isinstance(cm_dict,dict):
        return False
    if "cm" not in cm_dict.keys():
        return False
    cm = cm_dict['cm']
    if not isinstance(cm,np.ndarray):
        return False
    (cm_rows,cm_cols) = cm.shape
    if cm_rows != cm_cols:
        return False

    if "labels" not in cm_dict.keys():
        return False
    labels = cm_dict['labels']
    if len(labels) != cm_cols:
        return False

    return True

def make_cm_dict(cm: np.ndarray, labels: List[str]) -> dict:
    cm_dict = {
        'cm': cm, # numpy ndarray of shape (n_classes, n_classes)
        'labels': labels  # numpy ndarray of shape (n_classes,)
    }
    assert is_cm_dict(cm_dict), "DEBUG: cm_dict is not cm_dict"
    return cm_dict
