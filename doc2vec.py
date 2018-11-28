from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import check_file, NumpyEncoder
import json

path_data = './data/'
path_models = './models/'

def get_tagged_data(filename):
    tagged_data = []

    if check_file(path_data, filename):
        with open(path_data+filename) as f:
            for line in f:
                entry = json.loads(line)
                user = list(entry.keys())[0]
                tweets = entry[user]
                for tweet in tweets:
                    # data.append(tweet['tokens'])
                    td = TaggedDocument(words=tweet['tokens'], tags=[user+"_"+str(tweets.index(tweet))])
                    tagged_data.append(td)
    
    #tagged_data = [TaggedDocument(words=_d, tags=[user+"_"+str(i)]) for i, _d in enumerate(data)]
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
    model = Doc2Vec.load(model_name)
    #to find the vector of a document which is not in training data
    # test_data = ['short', 'data']
    # v1 = model.infer_vector(test_data)
    # print("V1_infer", v1)

    # to find most similar doc using tags
    # similar_doc = model.docvecs.most_similar('1')
    # print(similar_doc)


    #to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    #print(model.docvecs['VolumeBot'])
    # user_docs = [tag for tag in model.docvecs.offset2doctag if tag.startswith('VolumeBot')]
    # print(user_docs)
    # for doc_id in user_docs:
    #     print(model.docvecs[doc_id])



def user_embeddings(filename):
    def get_user_embedding(model, user):
        user_docs = [tag for tag in model.docvecs.offset2doctag if tag.startswith(user)]
        
        for doc_id in user_docs:
            if not user_docs.index(doc_id):
                user_vec = model.docvecs[doc_id]
            else:
                user_vec + model.docvecs[doc_id]
        
        user_vec /= len(user_docs)
        return user_vec

    def build_embeddings(model_name):
        model = Doc2Vec.load(model_name)
        user_embeddings = {}

        if check_file(path_data, filename):
            with open(path_data+filename) as f:
                for line in f:
                    user = list(json.loads(line).keys())[0]
                    user_vec = get_user_embedding(model, user)
                    user_embeddings[user] = user_vec

        return user_embeddings

    def write_combined_file(user_embeddings):
        #c_store = {k: set(v) for k, v in c_store.items()}
        print("Writing to file...")
        with open(path_data+filename[:-8]+"_EMBD.txt", 'w') as fp:
            for k, v in user_embeddings.items():
                json.dump({k:v}, fp, cls=NumpyEncoder)
                fp.write("\n")
        print("Write complete")

    embd_store = build_embeddings(path_models+'d2v.model')
    write_combined_file(embd_store)




tagged_data = get_tagged_data('st_comb_2018_01_01-2018_01_07_TKN.txt')
#print(tagged_data)
#train_model(tagged_data)

# #print(tagged_data, end="\n\n")
#analyse_model(path_models+'d2v.model')
user_embeddings('st_comb_2018_01_01-2018_01_07_TKN.txt')
# user_embeddings = build_embeddings(path_models+'d2v.model')
#print(get_user_embedding(path_models+'d2v.model'))


