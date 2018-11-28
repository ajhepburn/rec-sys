from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import check_file
import json

path = './data/'

def get_tagged_data(filename):
    data = []

    if check_file(path, filename):
        with open(path+filename) as f:
            first_entry = json.loads(f.readline())
            tweets = first_entry[list(first_entry.keys())[0]]
            for tweet in tweets:
                data.append(tweet['tokens'])
            
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data


get_tagged_data('st_comb_2018_01_01-2018_01_07_TKN.txt')


