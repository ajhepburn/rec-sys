from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec, KeyedVectors
from pathlib import Path
import numpy as np
import json, re

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
            model = KeyedVectors.load_word2vec_format(path+model_name, binary=True)
        else: raise Exception('Wrong type specified')
        
        return model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def date_prompt(self, dates):
    regexp = re.compile(r'\d{4}_\d{2}_\d{2}')

    print("\n"+"Please enter a FROM and TO date in the same format (ie. 2018_01_01):")
    date_from = input("FROM> ")
    date_to = input("TO> ")
    
    if date_from in dates and date_to in dates and regexp.match(date_to) and regexp.match(date_from):
        if date_to > date_from:
            return date_from, date_to
        else:
            print("TO date is earlier than FROM date.")
            return 0
    else:
        print("Dates do not exist in store, please try again.")
        return 0