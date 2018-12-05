from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import timedelta, datetime
from os import listdir
from os.path import isfile, join

from utils import check_file, NumpyEncoder, load_model, date_prompt
from analysis import Analysis, Doc2VecAnalysis

import numpy as np
import json, time


""" DOC2VEC RELATED FUNCTIONALITY

"""

class D2VTraining:
    def __init__(self, model_name, all=False):
        self.path_data = './data/'
        self.raw_path = self.path_data+'raw/'
        self.path_models = './models/'
        self.model_name = model_name
        self.all = all

    def query_dates(self):
        title = "TRAIN MODEL (DOC2VEC)"
        store = [f for f in listdir(self.raw_path) if isfile(join(self.raw_path, f)) and "stocktwits_messages_" in f]
        store.sort()
        
        if all:
            return store

        print("\n"+title+"\n"+("-"*len(title))+"\n"+"Data is available for the following dates:")
        dates = []
        for filename in store:
            dates.append(filename[-14:-4])
        print("{}, {}".format(", ".join(dates[:-1]), dates[-1]))

        date_from, date_to = None, None
        while date_from == None and date_to == None:
            try:
                date_from, date_to = date_prompt(self, dates)
            except TypeError: pass
        
        date_from_index, date_to_index = [i for i, s in enumerate(store) if date_from in s], [i for i, s in enumerate(store) if date_to in s]
        if date_from_index != None and date_to_index != None:
            files_selected = store[date_from_index[0]:date_to_index[0]+1]
        return files_selected

    def create_tagged_documents(self, filename):
        tagged_docs = []
        check_file(self.raw_path, filename)
        with open(self.raw_path+filename) as f:
            for line in f:
                entry = json.loads(line)
                user = list(entry.keys())[0]
                tweet = entry[user]
                td = TaggedDocument(words=tweet['tokens'], tags=[user])
                tagged_docs.append(td)
        f.close()
        return tagged_docs

    def train_model(self, tagged_docs):
        max_epochs = 30
        vec_size = 100
        # alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        min_count=2,
                        dm=0,
                        workers=8,
                        epochs=max_epochs)
        print("\nBuilding vocabulary started:", str(datetime.now()))
        vocab_start_time = time.monotonic()
        model.build_vocab(tagged_docs, progress_per=50000)
        vocab_end_time = time.monotonic()
        print("Building vocabulary ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=vocab_end_time - vocab_start_time))

        print("Training began:", str(datetime.now()))
        start_time = time.monotonic()
        model.train(tagged_docs,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
        # for epoch in range(max_epochs):
        #     print('iteration {0}'.format(epoch))
        #     model.train(tagged_docs,
        #                 total_examples=model.corpus_count,
        #                 epochs=model.epochs)
        #     # decrease the learning rate
        #     model.alpha -= 0.0002
        #     # fix the learning rate, no decay
        #     model.min_alpha = model.alpha
        end_time = time.monotonic()
        print("Training Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))

        model.save(self.path_models+self.model_name)
        print("Model Saved")

class D2VModel:
    def __init__(self, model):
        self.path_data = './data/'
        self.path_models = './models/'
        self.model = load_model(self.path_models, model, 'd2v')

    """ Functions below listed for writing to files.

    """
    
    def save_user_embeddings(self, obj):
        if len(list(obj.keys())) > 1:
            print("Writing to file...")
            with open(self.path_data+obj['ue_store_filename'][:-8]+"_EMBD.txt", 'w') as fp:
                for k, v in obj.items():
                    json.dump({k:v}, fp, cls=NumpyEncoder)
                    fp.write("\n")
            print("Write complete")
        else:
            print("Empty embeddings object passed as argument")

    """ Functions below are for user embedding calculation.
        build_user_embeddings_store calculates an average of all of the document vectors
        for a particular user, builds them to a store which can then be written to a file.

    """

    def get_user_embedding(self, user):
        user_doc_vecs = [self.model.docvecs[tag] for tag in self.model.docvecs.offset2doctag if tag.startswith(user)]
        user_vec = sum(user_doc_vecs)/len(user_doc_vecs)
        return user_vec

    def build_user_embeddings_store(self, filename):
        user_embeddings = {'ue_store_filename':filename}
        check_file(self.path_data, filename)
        print("User embeddings process began:", str(datetime.now()))
        start_time = time.monotonic()
        with open(self.path_data+filename) as f:
            for line in f:
                user = list(json.loads(line).keys())[0]
                user_vec = D2VModel.get_user_embedding(self, user)
                user_embeddings[user] = user_vec
        end_time = time.monotonic()
        print("User embeddings process ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))
        return user_embeddings

""" WORD2VEC RELATED FUNCTIONALITY

"""

class W2VModel:
    def __init__(self, model):
        self.model_path = './models/'
        self.model_name = model
        self.model = load_model(self.model_path, self.model_name, 'w2v')
        

# tagged_data = get_tagged_data('st_comb_2018_01_01-2018_01_07_TKN.txt')
# print(tagged_data)
# #train_model(tagged_data)

# # # #print(tagged_data, end="\n\n")
# # #analyse_model(path_models+'d2v.model')
# # user_embeddings('st_comb_2018_01_01-2018_01_07_TKN.txt')
# # # user_embeddings = build_embeddings(path_models+'d2v.model')
# # #print(get_user_embedding(path_models+'d2v.model'))

# model = D2VModel(model='d2v.model')
# t = Training('d2v.model')
# tagged_docs = t.create_tagged_documents('st_comb_2018_01_01-2018_01_07_TKN.txt')

# t = Training('d2v.model')
# tagged_documents = t.create_tagged_documents('st_comb_2018_01_01-2018_01_31_TKN.txt')
# t.train_model(tagged_documents)

# model = D2VModel(model='d2v.model')
# d2v_analysis = Doc2VecAnalysis(model='d2v.model', type='d2v')
# user_embeddings = model.build_user_embeddings_store('st_comb_2018_01_01-2018_01_31_TKN.txt')
# model.save_user_embeddings(user_embeddings)
#model.tsne('st_comb_2018_01_01-2018_01_31_TKN.txt')

# model = W2VModel(model='word2vec_twitter_model.bin')
# w2v_analysis = Analysis(model, 'w2v')
# print(w2v_analysis.get_vocab_size())

if __name__ == "__main__":
    t = D2VTraining(model_name='d2v_100d_dbow_2017ds.model', all=True)
    files = t.query_dates()
    tagged_docs = []
    print("\n")
    for f in files:
        tagged_docs += t.create_tagged_documents(f)
        print("Creating tagged documents... File Parsed: {0}, Total Documents: {1}".format(f, len(tagged_docs)), end="\r")
    t.train_model(tagged_docs)

        
