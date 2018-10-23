from gensim.models import Word2Vec

model_path = ('./models/')

model = Word2Vec.load(model_path+'model.bin')
words = list(model.wv.vocab)

count = 0
for word in words:
    if len(word) < 3:
        print(word)