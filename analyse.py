from gensim.models import Word2Vec

model_path = ('./models/')

model = Word2Vec.load(model_path+'model.bin')
words = list(model.wv.vocab)

for word, vocab_obj in model.wv.vocab.items():
    print(str(word) + str(vocab_obj.count))