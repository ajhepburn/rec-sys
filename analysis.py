from os import listdir
from os.path import isfile, join
from utils import check_file, load_model
from gensim.models import Word2Vec, Doc2Vec
import re, json
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, model_name, type):
        self.path_data = './data/'
        self.path_models = './models/'
        self.type = type
        self.model = load_model(self.path_models, model_name, type)

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
