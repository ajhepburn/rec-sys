from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
from pathlib import Path
import numpy as np
import json

def check_file(path, filename):
    my_file = Path(path+filename)
    try:
        my_file.resolve()
    #if my_file.is_file(): return 1
    except FileNotFoundError: raise

def load_model(path, model_name, type):
        check_file(path, model_name)
        if type == 'd2v':
            model = Doc2Vec.load(path+model_name)
        elif type == 'w2v':
            model = Word2Vec.load(path+'model_name')
        else: raise Exception('Wrong type specified')
        
        return model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)