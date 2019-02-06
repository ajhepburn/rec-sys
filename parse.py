import os, sys, json, spacy, pymongo, pprint, logging, re, io, glob, csv, logging
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
from itertools import islice
import dask.dataframe as dd

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
        self.watchlist = './data/csv/watchlist_clean.csv'

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

    def csv_write(self, wl_data):
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

    # def remove_null_byte_csv(self, f):
    #     ifile = os.path.join(self.wpath, f)
    #     with open (os.path.join(self.wpath, 'item_features_nonull.csv'), 'w') as item_f:
    #         headers = ['item_id', 'user_id', 'body', 'created_at', 'symbols']
    #         if_writer = csv.DictWriter(item_f, fieldnames=headers)
    #         if_writer.writeheader()

    #         data_initial = open(ifile, "rb")
    #         data = csv.reader((line.replace(b'\0','') for line in data_initial), delimiter=",")
    #         for row in data:
    #             if_writer.writerow(row)


    # def clean_items_csv(self, f):
    #     ifile = os.path.join(self.wpath, f)
    #     if not os.path.isfile(ifile):
    #         raise Exception("Missing CSV File")

    #     with open(ifile, 'r') as f:
    #         d_reader = csv.DictReader(f)
    #         headers = d_reader.fieldnames

    #     kept_cols = [i for i,x in enumerate(headers) if x in ('item_id', 'user_id', 'body', 'created_at', 'symbols')]

    #     df = dd.read_csv(ifile, usecols=kept_cols, engine='python', verbose =True , warn_bad_lines = True, error_bad_lines=False)
    #     # df = dd.read_csv(ifile, usecols=kept_cols, engine='c', quoting=csv.QUOTE_NONE, lineterminator='\n', error_bad_lines=False, encoding='utf8')
    #     df = df[df.symbols == df.symbols]
    #     df.to_csv('data/csv/ct_item_features-*.csv')

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
    wlp = WLParser()
    df = wlp.parse_wl()
    # df_wl = wlp.format_wl(df)
    ctp = CFParser('/media/ntfs/st_2017')
    ctp.csv_write(df)