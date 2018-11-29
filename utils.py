from gensim.models.doc2vec import Doc2Vec
from pathlib import Path
import numpy as np
import json

def check_file(path, filename):
    my_file = Path(path+filename)
    try:
        my_file.resolve()
    #if my_file.is_file(): return 1
    except FileNotFoundError: raise

def load_model(path, model_name):
        try:
            model = Doc2Vec.load(path+model_name)
        except FileNotFoundError: raise
        
        return model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)