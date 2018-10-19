from tokens import tokens
from gensim.models import Word2Vec
import time
from datetime import timedelta

sentences = tokens()
# train model
start_time = time.monotonic()
model = Word2Vec(sentences, min_count=25)
end_time = time.monotonic()
print("Time taken to train:",timedelta(seconds=end_time - start_time))

model.save('model.bin')
print(model)