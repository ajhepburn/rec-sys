from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import check_file
import json

path_data = './data/'
path_models = './models/'

def get_tagged_data(filename):
    data = []

    if check_file(path_data, filename):
        with open(path_data+filename) as f:
            first_entry = json.loads(f.readline())
            tweets = first_entry[list(first_entry.keys())[0]]
            for tweet in tweets:
                data.append(tweet['tokens'])
    
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(data)]
    return tagged_data

def train_model(tagged_data):
    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm=0)
    
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(path_models+"d2v.model")
    print("Model Saved")

def analyse_model(model_name):
    model= Doc2Vec.load(model_name)
    #to find the vector of a document which is not in training data
    # test_data = ['short', 'data']
    # v1 = model.infer_vector(test_data)
    # print("V1_infer", v1)

    # to find most similar doc using tags
    # similar_doc = model.docvecs.most_similar('1')
    # print(similar_doc)


    #to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    #print(model.docvecs['1'])

    #print(model.docvecs)


tagged_data = get_tagged_data('st_comb_2018_01_01-2018_01_07_TKN.txt')
#train_model(tagged_data)

print(tagged_data, end="\n\n")
analyse_model(path_models+'d2v.model')


