import os, sys, json, spacy, pymongo, pprint, logging, re, io
from datetime import timedelta, datetime
import pandas as pd

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

class CFParser:
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
            index = df_wl[(df_wl['user_id'] == row.user_id)].index.tolist()
            if not index:
                df_wl = df_wl.append({'user_id':row.user_id, ((content['group'], content['value'])):1}, ignore_index=True)
            else:
                if ((content['group'], content['value'])) not in df_wl:
                    df_wl[((content['group'], content['value']))] = 0
                # df_wl.ix[df_wl['user_id'] == row.user_id, ((content['group'], content['value']))] = 1
                df_wl.loc[df_wl['user_id'] == row.user_id, ((content['group'], content['value']))] = 1
        df_wl.fillna(0, inplace=True)
        return df_wl


if __name__ == "__main__":
    cfp = CFParser()
    #cfp.clean_csv()
    df = cfp.parse_wl()
    df_wl = cfp.format_wl(df)
