from os import listdir
from os.path import isfile, join
import time
from datetime import timedelta, datetime
import spacy, json, re
from utils import check_file

nlp = spacy.load('en')

# def tokens():
#     store = []
#     files = [f for f in listdir(path) if isfile(join(path, f))]

#     print("Tokenization began:", str(datetime.now()))
#     start_time = time.monotonic()
# #    for filename in nlp.pipe(files, batch_size=10000, n_threads=3):
#     for filename in files:
#         with open(path+filename) as f:
#             #head = [next(f) for x in range(5)]
#             for line in f:
#                 tokens = nlp(line.lower(), disable=['parser', 'tagger', 'ner'])
#                 tokens = [token.lemma_ for token in tokens if not token.orth_.isspace() and token.is_alpha and not token.is_stop and token.lemma_ != '-PRON-' and len(token.orth_) >= 3 and token.text in nlp.vocab]
#                 if len(tokens) != 0: store.append(tokens)

#     end_time = time.monotonic()
#     print("Tokenization Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))

#     return store

# class Tokenise():
#     def __init__(self):
#         self.path_data = './data/'

#     def tokenise

class Tokeniser:
    def __init__(self, filename):
        self.path_data = './data/'
        self.filename = filename
        
    def tokenise(self):
        check_file(self.path_data, self.filename)
        store = {}
        print("Tokenization began:", str(datetime.now()))
        start_time = time.monotonic()

        with open('./utilities/slang.txt') as sl:
            slang_terms = json.loads(sl.readline())
            slang_terms = [t.lower() for t in slang_terms]

        with open(self.path_data+self.filename) as f:
            for line in f:
                entry = json.loads(line)
                user = list(entry.keys())[0]
                tweets = entry[user]
                for tweet in tweets:
                    tweet_ct = re.sub(r'\$(\w+)',r'ZZZCASHTAGZZZ\1',tweet['body'])
                    tweet_ct = re.sub(r'&#39;s', r"", tweet_ct)
                    tweet_ct = re.sub(r'&#39;', r"'", tweet_ct)
                    tokens = nlp(tweet_ct.lower(), disable=['parser', 'tagger', 'ner'])
                    tokens = [token for token in tokens if not token.orth_.isspace() and token.is_alpha and not token.is_stop and token.orth_ not in slang_terms]
                    l_tokens = []
                    for token in tokens:
                        if token.orth_.startswith('zzzcashtagzzz'):
                            ct = token.text.replace(u'zzzcashtagzzz','$')
                            l_tokens.append(ct)
                        else:
                            l_token = token.lemma_
                            l_tokens.append(l_token)
                    #tokens = [x.replace(u'zzzcashtagzzz','$') for x in [token.text for token in iter(tokens)]]
                    #tokens = [token.lemma_ for token in tokens if not token.startswith('$')]
                    tokens = l_tokens
                    if len(tokens) > 3:
                        if user not in store:
                            store[user] = [{'id':tweet['id'], 'body':tweet['body'], 'tokens':tokens}]
                        else:
                            store[user].append({'id':tweet['id'], 'body':tweet['body'], 'tokens':tokens})

        end_time = time.monotonic()
        print("Tokenization Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))
        return store

    def write_to_file(self, store):
        print("Writing to file...")
        with open(self.path_data+self.filename[:-4]+"_TKN.txt", 'w') as fp:
            if store:
                for k, v in store.items():
                    json.dump({k:list(v)}, fp)
                    fp.write("\n")
        print("Write complete")

# def tokenize_combined_file(filename):
#     def write_combined_file(c_store):
#         #c_store = {k: set(v) for k, v in c_store.items()}
#         print("Writing to file...")
#         with open(path+filename[:-4]+"_TKN.txt", 'w') as fp:
#             for k, v in c_store.items():
#                 json.dump({k:list(v)}, fp)
#                 fp.write("\n")
#         print("Write complete")

#     def tokenize():
#         store = {}
#         print("Tokenization began:", str(datetime.now()))
#         start_time = time.monotonic()

#         with open('./utilities/slang.txt') as sl:
#             slang_terms = json.loads(sl.readline())
#             slang_terms = [t.lower() for t in slang_terms]

#         with open(path+filename) as f:
#             for line in f:
#                 entry = json.loads(line)
#                 user = list(entry.keys())[0]
#                 tweets = entry[user]
#                 for tweet in tweets:
#                     tweet_ct = re.sub(r'\$(\w+)',r'ZZZCASHTAGZZZ\1',tweet['body'])
#                     tweet_ct = re.sub(r'&#39;s', r"", tweet_ct)
#                     tweet_ct = re.sub(r'&#39;', r"'", tweet_ct)
#                     tokens = nlp(tweet_ct.lower(), disable=['parser', 'tagger', 'ner'])
#                     tokens = [token for token in tokens if not token.orth_.isspace() and token.is_alpha and not token.is_stop and token.orth_ not in slang_terms]
#                     tokens = [x.replace(u'zzzcashtagzzz','$') for x in [token.text for token in iter(tokens)]]
#                     if len(tokens) > 3:
#                         if user not in store:
#                             store[user] = [{'id':tweet['id'], 'body':tweet['body'], 'tokens':tokens}]
#                         else:
#                             store[user].append({'id':tweet['id'], 'body':tweet['body'], 'tokens':tokens})

#         end_time = time.monotonic()
#         print("Tokenization Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))
#         return store
#tokenize_combined_file('st_comb_2018_01_01-2018_01_07.txt')

#if check_file(path, filename):
tokeniser = Tokeniser('st_comb_2018_01_01-2018_01_31.txt')
store = tokeniser.tokenise()
tokeniser.write_to_file(store)

    

