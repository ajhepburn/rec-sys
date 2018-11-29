from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import Normalizer
from MulticoreTSNE import MulticoreTSNE as TSNE
from utils import check_file, NumpyEncoder, load_model
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import json, time

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
# from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

class Training:
    def __init__(self, model_name):
        self.path_data = './data/'
        self.path_models = './models/'
        self.model_name = model_name

    def create_tagged_documents(self, filename):
        tagged_docs = []
        check_file(self.path_data, filename)
        with open(self.path_data+filename) as f:
            for line in f:
                entry = json.loads(line)
                user = list(entry.keys())[0]
                tweets = entry[user]
                for tweet in tweets:
                    td = TaggedDocument(words=tweet['tokens'], tags=[user+"_"+str(tweets.index(tweet))])
                    tagged_docs.append(td)
        return tagged_docs

    def train_model(self, tagged_docs):
        max_epochs = 100
        vec_size = 20
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)
        
        model.build_vocab(tagged_docs)
        print("Training began:", str(datetime.now()))
        start_time = time.monotonic()
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_docs,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        end_time = time.monotonic()
        print("Training Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))

        model.save(self.path_models+self.model_name)
        print("Model Saved")

class D2VModel:
    def __init__(self, model):
        self.path_data = './data/'
        self.path_models = './models/'
        self.model = load_model(self.path_models, model)

    """ Functions below listed for writing to files.
    Visualisations like TSNE require normalisation and therefore rely on
    the save_doc_feature_vectors_normalised function. 

    """
    def save_doc_feature_vectors(self):
        np.save(self.path_data+'features/features-w2v-200.npy', self.model.docvecs.doctag_syn0)
    
    def save_doc_feature_vectors_normalised(self):
        nrm = Normalizer('l2')
        normed = nrm.fit_transform(self.model.docvecs.doctag_syn0)
        np.save('features/features_normed-w2v-200.npy',normed)
    
    def save_user_embeddings(self, obj, filename):
        print("Writing to file...")
        with open(self.path_data+filename[:-8]+"_EMBD.txt", 'w') as fp:
            for k, v in obj.items():
                json.dump({k:v}, fp, cls=NumpyEncoder)
                fp.write("\n")
        print("Write complete")

    """ Functions below are for user embedding calculation.
        build_user_embeddings_store calculates an average of all of the document vectors
        for a particular user, builds them to a store which can then be written to a file.

    """

    def get_user_embedding(self, user):
        user_docs = [tag for tag in self.model.docvecs.offset2doctag if tag.startswith(user)]
        for doc_id in user_docs:
            if not user_docs.index(doc_id):
                user_vec = self.model.docvecs[doc_id]
            else:
                user_vec + self.model.docvecs[doc_id]
        user_vec /= len(user_docs)
        return user_vec

    def build_user_embeddings_store(self, filename):
        user_embeddings = {}
        check_file(self.path_data, filename)
        with open(self.path_data+filename) as f:
            for line in f:
                user = list(json.loads(line).keys())[0]
                user_vec = D2VModel.get_user_embedding(self.model, user)
                user_embeddings[user] = user_vec
        return user_embeddings

    """ Functions below are responsible for textual analysis of the model.

    """

    def most_similar_documents(self, tag):
        similar_doc = self.model.docvecs.most_similar(tag)
        print(similar_doc)

    def print_vector_by_tag(self, tag):
        print(self.model.docvecs[tag])

    def print_vector_by_prefix(self, prefix_string):
        user_docs = [tag for tag in self.model.docvecs.offset2doctag if tag.startswith(prefix_string)]
        for doc_id in user_docs:
            print(self.model.docvecs[doc_id])

    def infer_vector(self, test_data_list):
        # to find the vector of a document which is not in training data
        v1 = self.model.infer_vector(test_data_list)
        print("V1_infer", v1)

    """ Functions below are responsible for visualising document vectors.
    t-SNE is used to show the most similar documents by embedding.

    """

    def tsne(self):
        #Creating and fitting the tsne model to the document embeddings
        tsne_model = TSNE(n_jobs=4, 
                        n_components=2,
                        verbose=1,
                        random_state=2018,
                        n_iter=300) #300
        tsne_d2v = tsne_model.fit_transform(self.model.docvecs.vectors_docs)

        #Putting the tsne information into sq
        tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])
        
        t = Training('d2v.model')
        tagged_docs = t.create_tagged_documents('st_comb_2018_01_01-2018_01_07_TKN.txt')

        tokens, tags = [], []
        for doc in tagged_docs:
            tokens.append(doc.words)
            tags.append(doc.tags)
        tokens, tags = pd.Series(tokens), pd.Series(tags)
        tsne_d2v_df['tokens'], tsne_d2v_df['tags'] = tokens.values, tags.values        

        plot_d2v = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Doc2vec document embeddings",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

        source = ColumnDataSource(data = dict(x = tsne_d2v_df["x"], 
                                            y = tsne_d2v_df["y"],
                                            tokens = tsne_d2v_df["tokens"],
                                            tags = tsne_d2v_df["tags"]))

        plot_d2v.scatter(x = "x", 
                        y = "y", 
                        source = source,
                        alpha = 0.7)
        hover = plot_d2v.select(dict(type = HoverTool))
        hover.tooltips = {"tokens": "@tokens", 
                        "tags": "@tags"}

        show(plot_d2v)

        


    





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

t = Training('d2v.model')
tagged_documents = t.create_tagged_documents('st_comb_2018_01_01-2018_01_31_TKN.txt')
t.train_model(tagged_documents)