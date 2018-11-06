from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
import re

def analyse_models(model_path):
    model = Word2Vec.load(model_path+'model.bin')
    words = list(model.wv.vocab)

    for word, vocab_obj in model.wv.vocab.items():
        print(str(word) + str(vocab_obj.count))

def analyse_tweet_bodies(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    regexp = re.compile(r'#\D+')

    for filename in files:
        with open(path+filename) as f:
            for line in f:
                if regexp.search(line):
                    print(line)


#analyse_models('./models/')
analyse_tweet_bodies("./data/")