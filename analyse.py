from os import listdir
from os.path import isfile, join
from utils import check_file, load_model
from gensim.models import Word2Vec, Doc2Vec
from sklearn.manifold import TSNE
import re, json
import numpy as np
import matplotlib.pyplot as plt 

path_models = './models/'
path_data = './data/'

# def analyse_models(model_path):
#     model = Word2Vec.load(model_path+'model.bin')
#     words = list(model.wv.vocab)

#     for word, vocab_obj in model.wv.vocab.items():
#         print(str(word) + str(vocab_obj.count))

# def analyse_tweet_bodies(path):
#     files = [f for f in listdir(path) if isfile(join(path, f))]
#     regexp = re.compile(r'#\D+')

#     for filename in files:
#         with open(path+filename) as f:
#             for line in f:
#                 if regexp.search(line):
#                     print(line)

def analyse_models():
    # def load_model():
    #     try:
    #         model = Doc2Vec.load(path_model+model_name)
    #     except FileNotFoundError: raise
        
    #     return model
    model = load_model(path_models, 'd2v.model')



def analyse_users(filename):
    def build_object():
        try:
            with open(path_data+filename) as f:
                data = {}
                print("Building user object...", end="\n\n")
                for line in f:
                    user_data = json.loads(line)
                    username = list(user_data.keys())[0]
                    data[username] = user_data[username]
        except FileNotFoundError: raise

        return data

    def get_user_tweets(user):
        data = build_object()     
        if user in data:
            line = "Tweets for: "+ user
            print(line)
            print("-"*len(line), end="\n\n")
            tweets = data[user]
            for tweet in tweets:                    
                print("ID:", tweet['id'])
                print("Body:", tweet['body'])
                print("Tokens:", tweet['tokens'], end="\n\n")
        else:
            print("User not in data.")



#analyse_models('./models/')
#analyse_tweet_bodies("./data/")
# analyse_users('st_comb_2018_01_01-2018_01_07_TKN.txt')
analyse_models()