import os, sys, json, spacy, pymongo, pprint, logging, re, io, glob, csv, logging, codecs
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import dask.dataframe as dd

class CBParser:
    def __init__(self, fp: str):
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

    def tokenize(self, tweet: str) -> list:
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

class CFParser:
    def __init__(self, fp: str):
        self.rpath = fp
        self.wpath = './data/csv/'
        self.logpath = './log/io/'
        self.watchlist = self.wpath+'watchlist_clean.csv'
        
    def parse_wl(self) -> pd.core.frame.DataFrame:
        pd.set_option('display.max_colwidth', -1)
        df0 = pd.read_csv(self.watchlist, sep=";", header=None)
        df = df0[0].str.split(';',expand=True)
        df.columns = ["user_id", "content"]
        return df

    def csv_write(self, wl_data: pd.core.frame.DataFrame):
        if os.path.exists(os.path.join(self.wpath, 'user_features.csv')) and os.path.exists(os.path.join(self.wpath, 'user_features.csv')): raise Exception("User/Item Features Already Exist! Exiting...")
        logging.basicConfig(filename=os.path.join(self.logpath, str(datetime.now())[:-7]+'_log (cashtag_csv).log'),level=logging.INFO)
        logging.info("Starting CSV write at "+str(datetime.now())[:-7])
        with open (os.path.join(self.wpath, 'user_features.csv'), 'w', encoding='UTF-8', newline='') as user_f, open (os.path.join(self.wpath, 'item_features.csv'), 'w', encoding='UTF-8', newline='') as item_f:
            item_fields, user_fields = ('item_id', 'user_id', 'body', 'created_at', 'ct_id', 'ct_symbol', 'ct_title', 'ct_exchange','ct_sector','ct_industry','ct_trending_score', 'ct_watchlist_count'), ('id', 'username', 'join_date', 'followers', 'following', 'ideas', 'like_count', 'subscribers_count', 'subscribed_to_count', 'location', 'wt_id', 'wt_type', 'wt_group', 'wt_value', 'wt_display')
            uf_writer, if_writer = csv.DictWriter(user_f, fieldnames=user_fields, delimiter="\t"), csv.DictWriter(item_f, fieldnames=item_fields, delimiter="\t")
            uf_writer.writeheader()
            if_writer.writeheader()

            for filepath in glob.glob(os.path.join(self.rpath, '*.json')):
                with open(filepath) as f:
                    logging.info("Read: "+filepath)
                    for line in f:
                        content = json.loads(line)
                        item_in = content['data']
                        if 'symbols' not in item_in:
                            continue
                        user_in = content['data'].pop('user')
                        
                        wl = wl_data['content'][wl_data['user_id'].astype(str) == str(user_in['id'])].values
                        if len(wl) > 0: 
                            for wl_item in wl:
                                user = {k: user_in[k] for k in user_fields if k in user_in}
                                u_headers = [item[3:] for item in user_fields if item.startswith('wt_')]
                                for k in u_headers:
                                    item = json.loads(wl_item[1:-1])
                                    user['wt_'+k] = (item['group'], item['value']) if k == 'id' else item[k]
                        else:
                            user = {k: user_in[k] for k in user_fields if k in user_in}
                        uf_writer.writerow(user)
                        
                        item_in['item_id'] = item_in.pop('id')
                        item_in['user_id'] = user_in['id']

                        for symbol in item_in['symbols']:
                            item = {k: item_in[k] for k in item_fields if k in item_in}
                            i_headers = [item[3:] for item in item_fields if item.startswith('ct_')]
                            for k in i_headers:
                                item["ct_"+k] = symbol[k]
                                item['body'] = item['body'].replace('\t',' ')
                                if_writer.writerow(item)

        logging.info("Finished CSV write at "+str(datetime.now())[:-7])

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
    # def parse_interactions(self):
    #     # item_features = csv.DictReader(open(os.path.join(self.wpath, 'item_features.csv')), delimiter='\t')

    #     if os.path.exists(os.path.join(self.wpath, 'interaction_data.csv')): raise Exception('Interaction Data Already Exists, Exiting...')
    #     logging.basicConfig(filename=os.path.join(self.logpath, str(datetime.now())[:-7]+'_log (interactions_csv).log'),level=logging.INFO)
    #     logging.info("Starting CSV write at "+str(datetime.now())[:-7])
    #     with open(os.path.join(self.wpath, 'item_features.csv'), 'rU') as item_features, open (os.path.join(self.wpath, 'interaction_data.csv'), 'w', encoding='UTF-8', newline='') as idata:
    #         fields = ('user_id', 'item_id')
    #         writer = csv.DictWriter(idata, fieldnames=fields, delimiter="\t")
    #         writer.writeheader()

    #         # for entry in item_features:
    #         for row in csv.DictReader((line.replace('\0','') for line in item_features), delimiter="\t"):
    #             writer.writerow({'user_id':row['user_id'], 'item_id':row['item_id']})

    #     logging.info("Finished CSV write at "+str(datetime.now())[:-7])
    #     for handler in logging.root.handlers[:]:
    #         logging.root.removeHandler(handler)

    def parse_stocktwits_data(self):
        logging.basicConfig(filename=os.path.join(self.logpath, str(datetime.now())[:-7]+'_log (stocktwits_csv).log'),level=logging.INFO)
        logging.info("Starting CSV write at "+str(datetime.now())[:-7])

        with open (os.path.join(self.wpath, 'stocktwits.csv'), 'w', newline='') as stocktwits_csv:
            fields = ['user_id', 'item_id', 'item_cashtags']
            writer = csv.DictWriter(stocktwits_csv, fieldnames=fields, delimiter='\t')
            writer.writeheader()

            for filepath in glob.glob(os.path.join(self.rpath, "*")):
                with open(filepath) as f:
                    if filepath.endswith('2017_02_01'): break
                    logging.info("Read: "+filepath)
                    for line in f:
                        content = json.loads(line)['data']
                        if 'symbols' not in content: continue
                        user_id = content['user']['id']
                        item_id = content['id']
                        cashtags = []
                        for symbol in content['symbols']:
                            cashtags.append(symbol['symbol'])
                        writer.writerow({'user_id':user_id, 'item_id':item_id, 'item_cashtags':cashtags})

        logging.info("Finished CSV write at "+str(datetime.now())[:-7])

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)


if __name__ == "__main__":
    cfp = CFParser('/media/ntfs/st_2017/')
    cfp.parse_stocktwits_data()
    # cfp.parse_interactions()