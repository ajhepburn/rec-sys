from os import listdir
from os.path import isfile, join
from utils import check_file, load_model
from gensim.models import Word2Vec, Doc2Vec
import re, json
import numpy as np
import matplotlib.pyplot as plt 

class Analysis:
    def __init__(self, model, type):
        self.path_data = './data/'
        self.path_models = './models/'
        self.type = type
        self.model = load_model(self.path_models, model, self.type)

    def infer_vector(self, test_data_list):
        # to find the vector of a document which is not in training data
        v1 = self.model.infer_vector(test_data_list)
        print("V1_infer", v1)

    def get_vocab_size(self):
        return len(self.model.wv.vocab)

    def most_similar_words(self, word):
        similar=self.model.wv.most_similar(word)
        words=list((w[0] for w in similar))
        return words

    def subtract_from_vectors(self, term1, term2, term_to_remove):
        similar=self.model.wv.most_similar(positive=[term1, term2], negative=[term_to_remove], topn=1)
        words=list((w[0] for w in similar))
        return words

class Doc2VecAnalysis(Analysis):
    def get_number_of_docs(self):
        return len(self.model.docvecs)

    def most_similar_documents(self, tag):
        similar_doc = self.model.docvecs.most_similar(tag)
        return similar_doc

    def print_vector_by_tag(self, tag):
        return self.model.docvecs[tag]

    def print_vector_by_prefix(self, prefix_string):
        user_docs = [self.model.docvecs[tag] for tag in self.model.docvecs.offset2doctag if tag.startswith(prefix_string)]
        # for doc_id in user_docs:
        #     print(self.model.docvecs[doc_id])
        return user_docs

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

# def analyse_users(filename):
#     def build_object():
#         try:
#             with open(path_data+filename) as f:
#                 data = {}
#                 print("Building user object...", end="\n\n")
#                 for line in f:
#                     user_data = json.loads(line)
#                     username = list(user_data.keys())[0]
#                     data[username] = user_data[username]
#         except FileNotFoundError: raise

#         return data

#     def get_user_tweets(user):
#         data = build_object()     
#         if user in data:
#             line = "Tweets for: "+ user
#             print(line)
#             print("-"*len(line), end="\n\n")
#             tweets = data[user]
#             for tweet in tweets:                    
#                 print("ID:", tweet['id'])
#                 print("Body:", tweet['body'])
#                 print("Tokens:", tweet['tokens'], end="\n\n")
#         else:
#             print("User not in data.")



#analyse_models('./models/')
#analyse_tweet_bodies("./data/")
# analyse_users('st_comb_2018_01_01-2018_01_07_TKN.txt')
#analyse_models()