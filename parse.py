import os, sys, json, spacy, pymongo, pprint, logging, re
from datetime import timedelta, datetime

class Parser:
    def __init__(self, fp):
        if not os.path.isdir(fp):
            raise OSError('ERROR: Directory does not exist.')
        self.path = fp
        self.files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        self.files.sort()
        with open('./utilities/slang.txt') as sl:
            slang_terms = json.loads(sl.readline())
            self.slang_terms = [t.lower() for t in slang_terms]
        self.nlp = spacy.load('en')
        self.log_path = './log/'

    def test_db_conn(self):
        pass

    def tokenize(self, tweet):
        tweet_ct = re.sub(r'\$(\w+)',r'ZZZCASHTAGZZZ\1',tweet)
        tweet_ct = re.sub(r'&#39;s', r"", tweet_ct)
        tweet_ct = re.sub(r'&#39;', r"'", tweet_ct)
        tokens = self.nlp(tweet_ct.lower(), disable=['parser', 'tagger', 'ner'])
        tokens = [token for token in tokens if not token.orth_.isspace() and re.match('^[a-z]*$', token.orth_) and not token.is_stop and token.orth_ not in self.slang_terms and 'haha' not in token.orth_ and token.lemma_ != '-PRON-' and len(token.orth_) > 3]
        l_tokens = []
        for token in tokens:
            if token.orth_.startswith('zzzcashtagzzz'):
                ct = token.text.replace(u'zzzcashtagzzz','$')
                l_tokens.append(ct)
            else:
                l_token = token.lemma_
                l_tokens.append(l_token)
        
        tokens = []
        for token in l_tokens:
            if not token.startswith('$'):
                token = re.sub(r'(.)\1+', r'\1\1', token)
                tokens.append(token)
            else:
                tokens.append(token)
        return tokens if len(tokens) > 2 else []

    def extract_features(self, inputFile):
        pass

    def build_wlist(self):
        pass

    def db_insert(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["recsys"]
        col = db["messages"]
        # bulk = col.initialize_ordered_bulk_op()

        logging.basicConfig(filename=self.log_path+'database/insertion_'+str(datetime.now())+'.log',level=logging.INFO)
        
        for filename in self.files:
            logging.info(str(datetime.now())+" - File: "+filename)
            with open(self.path+filename) as inputFile:  
                for line in inputFile:
                    data = json.loads(line)
                    # tokens = self.tokenize(data['data']['body'])
                    # data['data']['tokens'] = tokens
                    user = data['data'].pop('user')
                    # user_id = user['id']
                    # query = col.find({"id": user_id}, {"id": 1}).limit(1).count()
                    
                    # user['messages'] = [data]

                    # bulk.find({'id':user['id']}).upsert().update({ '$setOnInsert': {'username':user['username'], 'name':user['name'], 'join_date':user['join_date'], 'classification':user['classification'], 'followers':user['followers'], 'following':user['following'], 'ideas':user['ideas'], 'like_count':user['like_count'], 'subscribers_count':user['subscribers_count'], 'following_stocks':user['following_stocks'], 'location':user['location'], 'bio':user['bio'], 'trading_strategy':user['trading_strategy'], 'messages':[data]}})
                    col.update_one({'id':user['id']},{ '$set': {'username':user['username'], 'name':user['name'], 'join_date':user['join_date'], 'classification':user['classification'], 'followers':user['followers'], 'following':user['following'], 'ideas':user['ideas'], 'like_count':user['like_count'], 'subscribers_count':user['subscribers_count'], 'following_stocks':user['following_stocks'], 'location':user['location'], 'bio':user['bio'], 'trading_strategy':user['trading_strategy']}, '$addToSet':{'messages':data}}, upsert=True)

                    # if not query:
                    #     user['messages'] = [data]
                    #     col.insert_one(user)
                    # else:
                    #     col.update_one({'id': user_id}, {'$push': {'messages': data}})
                    
                    
                    

        # CLEAR LOG CONFIGS
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

if __name__ == "__main__":
    pd = Parser("/media/ntfs/st_2017/")
    pd.db_insert()
