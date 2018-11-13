from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
import re, json

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

def analyse_tokens(file):
    with open('./data/'+file) as data:
        for raw_line in data:
            line = json.loads(raw_line)
            user = list(line.keys())[0]
            tweets = line[user]
            for tweet in tweets:
                body = tweet['body']
                tokens = tweet['tokens']
                print(user, tokens)


#analyse_models('./models/')
#analyse_tweet_bodies("./data/")
analyse_tokens('st_comb_2018_01_01-2018_01_07_TKN.txt')