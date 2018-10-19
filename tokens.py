from os import listdir
from os.path import isfile, join
import time
from datetime import timedelta
import spacy
nlp = spacy.load('en_core_web_sm')

def tokens(path):
    store = []
    files = [f for f in listdir(path) if isfile(join(path, f))]

    start_time = time.monotonic()
    for filename in files:
        with open("./data/"+filename) as f:
            #head = [next(f) for x in range(5)]
            for line in f:
                tokens = nlp(line.lower())
                tokens = [token.lemma_ for token in tokens if not token.orth_.isspace() and token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-']
                store.append(tokens)

    end_time = time.monotonic()
    print("Time taken to tokenize:",timedelta(seconds=end_time - start_time))

    return store