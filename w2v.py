from tokens import tokens
from gensim.models import Word2Vec
import time
from datetime import timedelta, datetime

model_path = './models/'

sentences = tokens()
# train model
print("Training began:", str(datetime.now()))
start_time = time.monotonic()
model = Word2Vec(sentences, min_count=25, size=200, sg=1)
end_time = time.monotonic()
print("Training ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))

model.save(model_path+'model.bin')
print(model)