import os
import datetime
from tensorflow.keras import models

def save_model(models_root_dir, model):
    assert model is not None
    assert models_root_dir is not None
    '''saves the given model to a unique model_dir_path, and return the path'''
    dt = datetime.datetime.utcnow().isoformat()
    model_dir_path = os.path.join(models_root_dir, f"model-{dt}")
    model.save(model_dir_path)
    return model_dir_path

def find_latest_model_dir_path(models_root_dir):
    '''find the latest model_dir_path under models_root_dir'''

    f_paths = [os.path.join(models_root_dir,f) for f in os.listdir(models_root_dir) if f.startswith('model-')]
    model_dir_paths = [f_path for f_path in f_paths if  os.path.isdir(f_path) ]
    
    if len(model_dir_paths) == 0:
        print(f"no model_dir_paths found under {models_root_dir}")
        return None
    
    latest_model_dir_path = None
    max_ctime = 0.0
    for model_dir_path in model_dir_paths:
        ctime = os.path.getctime(model_dir_path)
        if ctime > max_ctime:
            latest_model_dir_path = model_dir_path
    assert latest_model_dir_path is not None
    return latest_model_dir_path

def load_model(model_dir_path):
    '''load the model from the given model_dir_path'''
    if model_dir_path is not None and os.path.isdir(model_dir_path):
        model = models.load_model(model_dir_path)
        assert model is not None
        return model
    return None

def load_latest_model(models_root_dir):
    model_dir_path = find_latest_model_dir_path(models_root_dir)
    if model_dir_path is not None:
        return load_model(model_dir_path)
    return None

def get_md5_hash(file_path):
    import hashlib
    with open(file_path) as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()

def model_files_are_equivalent(model_dir_path_A, model_dir_path_B):
    md5_A = get_md5_hash(os.path.join(model_dir_path_A,"saved_model.pb"))
    md5_B = get_md5_hash(os.path.join(model_dir_path_B,"saved_model.pb"))
    return md5_A == md5_B

def models_are_equivalent(modelA,modelB):
    typesA = [type(layer) for layer in modelA.layers]
    typesB = [type(layer) for layer in modelB.layers]
    N = len(typesA)
    if len(typesB) != N:
        return False
    for i in range(N):
        if typesA[i] != typesB[i]:
            return False
    return True

