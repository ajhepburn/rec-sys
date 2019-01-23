# from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

from utils import load_model, check_file
from models import D2VTraining

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
# from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

class Plotting:
    def __init__(self, model, type):
        self.path_data = './data/'
        self.path_models = './models/'
        self.type = type
        self.model = load_model(self.path_models, model, self.type)

    def tsne_w2v(self, word):
        if not self.type == 'w2v': raise ValueError('Only works with word2vec models')
        arr = np.empty((0,200), dtype='f')
        word_labels = [word]

        # get close words
        close_words = self.model.similar_by_word(word)
        
        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([self.model[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = self.model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
            
        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        # display scatter plot
        plt.scatter(x_coords, y_coords)

        for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
        plt.show()

    def tsne_d2v(self, filename):
        if not self.type == 'd2v': raise ValueError('Only works with doc2vec models')
        check_file(self.path_data, filename)
        #Creating and fitting the tsne model to the document embeddings
        tsne_model = TSNE(n_jobs=4, 
                        n_components=2,
                        verbose=1,
                        random_state=2018,
                        n_iter=300) #300
        tsne_d2v = tsne_model.fit_transform(self.model.docvecs.vectors_docs)

        #Putting the tsne information into sq
        tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])
        
        t = D2VTraining('d2v.model')
        tagged_docs = t.create_tagged_documents(filename)

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

    def pca_w2v(self):
        if not self.type == 'w2v': raise ValueError('Only works with word2vec models')
        # fit a 2d PCA model to the vectors
        X = self.model[self.model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        plt.scatter(result[:, 0], result[:, 1])
        words = list(self.model.wv.vocab)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        plt.show()