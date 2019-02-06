import os, sys, json, spacy, pymongo, pprint, logging, re, io, glob, csv, logging
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
from itertools import islice

class CBParser:
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

class WLParser:
    def __init__(self):
        self.watchlist = './data/watchlist_clean.csv'

    def clean_csv(self):
        df = pd.read_csv('./data/watchlists.csv', sep="\t", header=None)
        pd.set_option('display.max_colwidth', -1)
        df=df.replace({"&#39;":"'"}, regex=True)
        # df[0] = df[0].apply(lambda x:str(x).replace(re.match('&#39;'),"'"))
        df.to_csv('./data/watchlist_clean.csv', index=False, header=None)
        # print(df[0].str.extract(r'(&#39;)', expand=False))

    def parse_wl(self):
        pd.set_option('display.max_colwidth', -1)
        df0 = pd.read_csv(self.watchlist, sep=";", header=None)
        df = df0[0].str.split(';',expand=True)
        df.columns = ["user_id", "content"]
        return df

    def format_wl(self, df):
        df_wl = pd.DataFrame(columns=['user_id'])
        for row in df.itertuples():
            content = json.loads(row.content[1:-1])
            df_wl = df_wl.append({'user_id':row.user_id, 'item_id':((content['group'], content['value']))}, ignore_index=True)
        return df_wl

class CFParser:
    def __init__(self, fp):
        self.rpath = fp
        self.wpath = './data/csv/'
        self.logpath = './log/io/'
        # self.files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        # self.files.sort()

    def csv_write(self):
        logging.basicConfig(filename=os.path.join(self.logpath, str(datetime.now())[:-7]+'_log (cashtag_csv).log'),level=logging.INFO)
        logging.info("Starting CSV write at "+str(datetime.now())[:-7])
        with open (os.path.join(self.wpath, 'user_features.csv'), 'w') as user_f, open (os.path.join(self.wpath, 'item_features.csv'), 'w') as item_f:
            user_fieldnames, item_fieldnames = ['id', 'username', 'name', 'avatar_url', 'avatar_url_ssl', 'join_date', 'official', 'identity', 'classification', 'followers', 'following', 'ideas', 'watchlist_stocks_count', 'like_count', 'subscribers_count', 'subscribed_to_count', 'following_stocks', 'location', 'bio', 'website_url', 'trading_strategy'], ['item_id', 'user_id', 'body', 'created_at', 'source', 'conversation', 'symbols', 'prices', 'reshares', 'mentioned_users', 'entities', 'links', 'sentiment', 'network','structurable','reshare_message', 'likes', 'disclosure']
            uf_writer, if_writer = csv.DictWriter(user_f, fieldnames=user_fieldnames), csv.DictWriter(item_f, fieldnames=item_fieldnames)
            uf_writer.writeheader()
            if_writer.writeheader()

            for filepath in glob.glob(os.path.join(self.rpath, '*.json')):
                with open(filepath) as f:
                    logging.info("Read: "+filepath)
                    for line in f:
                        content = json.loads(line)
                        item = content['data']
                        user = content['data'].pop('user')
                        
                        item['item_id'] = item.pop('id')
                        item['user_id'] = user['id']
                        
                        uf_writer.writerow(user)
                        if_writer.writerow(item)
        logging.info("Finished CSV write at "+str(datetime.now())[:-7])

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    def extract_features(self, feature):
        
        if not feature == 'user' and not feature == 'item':
            raise Exception("Unrecognised Features Type: "+feature)
        
        ffile = os.path.join(self.wpath, feature+'_features.csv')
        if not os.path.isfile(ffile):
            raise Exception("Missing Features File")

        def extract_user_features(dr):
            for line in islice(dr, 1):
                print(json.dumps(line, indent=4))

        def extract_item_features(dr):
            for line in islice(dr, 1):
                print(json.dumps(line, indent=4))
        
        dr = csv.DictReader(open(ffile),delimiter=",")
        locals()["extract_"+feature+"_features"](dr)

                        

                



if __name__ == "__main__":
    # wlp = WLParser()
    # df = wlp.parse_wl()
    # df_wl = wlp.format_wl(df)
    ctp = CFParser('/media/ntfs/st_2017')
    ctp.extract_features('item')