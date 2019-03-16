from datetime import datetime

from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

from os.path import join

import pandas as pd
import numpy as np
import os, json, sys, logging, csv


class AttributeParser:
    def __init__(self, limit: str):
        self.rpath = '/media/ntfs/st_2017'
        self.wpath = './data/csv/'
        self.logpath = './log/io/csv/'

        self.files = [f for f in os.listdir(self.rpath) if os.path.isfile(os.path.join(self.rpath, f))]
        self.files.sort()
        self.files = self.files[:next(self.files.index(x) for x in self.files if x.endswith(limit))]

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1}_log ({2}).log".format(self.logpath, str(datetime.now())[:-7],'metadata_csv')),
                    logging.StreamHandler(sys.stdout)
                ])
    
    def parse(self, l) -> tuple:
        """ Takes in a single line and attempts to retrieve user and item information for feature engineering.
        
        Will return a tuple filled with None values if:
            * No cashtags are found with at least one industry tag.
        If there are no cashtags at all in a single tweet, the line will be skipped.

        Returns:
            tuple: Returns a tuple of User ID, User location data, Item ID, Item Timestamp, Item Body, Cashtag Titles, Cashtag Industries, Cashtag Sectors

        """

        d = json.loads(l)['data']
        symbols = d.get('symbols', False)
        arr = (titles, tag_ids, cashtags, trending_scores, watchlist_counts, exchanges, industries, sectors) = [], [], [], [], [], [], [], []

        if symbols:                     # Checks to see if cashtags exist in tweet
            for s in symbols:
                check = (s_id, s_title, s_symbol, s_trending_score, s_watchlist_count, s_exchange, s_industry, s_sector) = s.get('id'), s.get('title'), s.get('symbol'), s.get('trending_score'), s.get('watchlist_count'), s.get('exchange'), s.get('industry'), s.get('sector')  # Check to see if a cashtag contains an 'Industry' tag, otherwise skip
                if not all(check):
                    continue

                user_id, user_location = d.get('user').get('id'), d.get('user').get('location')
                item_id, item_timestamp, item_body = d.get('id'), d.get('created_at'), d.get('body').replace('\t',' ').replace('\n','')

                titles[len(titles):], tag_ids[len(tag_ids):], cashtags[len(cashtags):], trending_scores[len(trending_scores):], watchlist_counts[len(watchlist_counts):], exchanges[len(exchanges):], industries[len(industries):], sectors[len(sectors):] = tuple(zip((s_title, s_id, s_symbol, s_trending_score, s_watchlist_count, s_exchange, s_industry, s_sector)))

        return (user_id, user_location, item_id, item_timestamp, item_body, titles, tag_ids, cashtags, trending_scores, watchlist_counts, exchanges, industries, sectors) if all(arr) else ((None),)*13
                #return out_dict
                 
    def file_writer(self) -> int:
        """ Responsible for writing to ct_industry.csv in ./data/csv/ and logging each file read.
        
        Passes single line into self.parse which returns a tuple of metadata, this is then written
        to a single row in the CSV, provided the tuple returned by self.parse does not contain any None values.

        Returns:
            int: Returns the number of documents in the file.

        """

        logger = logging.getLogger()
        with open (os.path.join(self.wpath, 'metadata.csv'), 'w', newline='') as stocktwits_csv:
            fields = ['user_id', 'item_id', 'user_location', 'item_timestamp', 'item_body','item_titles','item_tag_ids','item_cashtags','item_tag_trending_score','item_tag_watchlist_count','item_exchanges','item_industries','item_sectors']
            writer = csv.DictWriter(stocktwits_csv, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            line_count = 0

            for fp in self.files:
                logger.info('Reading file {0}'.format(fp))
                with open(os.path.join(self.rpath, fp)) as f:
                    for l in f:
                        if not all(self.parse(l)):
                            continue
                        user_id, user_location, item_id, item_timestamp, item_body, titles, item_tag_ids, cashtags, trend_scores, watch_counts, exchanges, industries, sectors = self.parse(l)
                        writer.writerow({'user_id':user_id, 'item_id':item_id, 'user_location':user_location, 'item_body':item_body,'item_timestamp':item_timestamp, 'item_titles':'|'.join(map(str, titles)),'item_tag_ids':'|'.join(map(str, item_tag_ids)),'item_cashtags':'|'.join(map(str, cashtags)),'item_tag_trending_score':'|'.join(map(str, trend_scores)), 'item_tag_watchlist_count':'|'.join(map(str, watch_counts)), 'item_exchanges':'|'.join(map(str, exchanges)),'item_industries':'|'.join(map(str, industries)),'item_sectors':'|'.join(map(str, sectors))})
                        line_count+=1
        
        return line_count

    def run(self):
        self.logger()
        logger = logging.getLogger()
        logger.info("Starting CSV write")

        line_count = self.file_writer()

        logger.info("Finished CSV write with {0} documents".format(line_count))

class CashtagParser:
    def __init__(self):
        self.rpath = './data/csv/metadata_clean.csv'
        self.wpath = './data/csv/cashtags_clean.csv'

    def conversion_to_cashtag_orientation(self):
        def explode(df, lst_cols, fill_value='', preserve_index=False):
            # make sure `lst_cols` is list-alike
            if (lst_cols is not None
                and len(lst_cols) > 0
                and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
                lst_cols = [lst_cols]
            # all columns except `lst_cols`
            idx_cols = df.columns.difference(lst_cols)
            # calculate lengths of lists
            lens = df[lst_cols[0]].str.len()
            # preserve original index values    
            idx = np.repeat(df.index.values, lens)
            # create "exploded" DF
            res = (pd.DataFrame({
                        col:np.repeat(df[col].values, lens)
                        for col in idx_cols},
                        index=idx)
                    .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                                    for col in lst_cols}))
            # append those rows that have empty lists
            if (lens == 0).any():
                # at least one list in cells is empty
                res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                        .fillna(fill_value))
            # revert the original index order
            res = res.sort_index()
            # reset index if requested
            if not preserve_index:        
                res = res.reset_index(drop=True)
            return res

        df = pd.read_csv(self.rpath, sep='\t')
        df = df.drop(['user_loc_check', 'user_token_check'], axis=1)
        for el in ('item_titles','item_tag_ids','item_cashtags', 'item_tag_trending_score','item_tag_watchlist_count', 'item_industries', 'item_sectors', 'item_exchanges'):
            df[el] = df[el].apply(lambda x: x.split('|') if '|' in x else [x])
        df0 = explode(df, ['item_titles','item_tag_ids', 'item_cashtags','item_tag_trending_score','item_tag_watchlist_count','item_exchanges','item_industries', 'item_sectors'], fill_value='')
        
        df0['item_tag_trending_score'], df0['item_tag_watchlist_count'] = df0['item_tag_trending_score'].astype(float), df0['item_tag_watchlist_count'].astype(int)
        df0['item_tag_trending_score'] = (df0['item_tag_trending_score'] - min(df0['item_tag_trending_score']))/(max(df0['item_tag_trending_score']) - min(df0['item_tag_trending_score']))
        df0['item_tag_watchlist_count'] = (df0['item_tag_watchlist_count'] - min(df0['item_tag_watchlist_count']))/(max(df0['item_tag_watchlist_count']) - min(df0['item_tag_watchlist_count']))
        df0.to_csv(self.wpath, sep='\t')

class FMParser:
    def __init__(self):
        self._modelpath = './models/xdeepfm/'
        self._logpath = './log/models/xdeepfm/'
        self._rpath = './data/csv/cashtags_clean.csv'

    def logger(self):
            """Sets logger config to both std.out and log ./log/models/smartadaptiverec/

            Also sets the formatting instructions for the log file, prints Time,
            Current Thread, Logging Type, Message.

            """
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} ({2}).log".format(
                        self._logpath,
                        str(datetime.now())[:-7],
                        'smart_adaptive_rec',
                    )),
                    logging.StreamHandler(sys.stdout)
                ])

    def csv_to_df(self, months: int) -> tuple:
        """Reads in CSV file, converts it to a number of Pandas DataFrames.

        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and
                interactions between items.

        """

        df = pd.read_csv(self._rpath, sep='\t')
        df['count'] = df.groupby(['user_id', 'item_tag_ids']).user_id.transform('size')
        df = df[df['count'] < months*100]
        df_weights = df[['user_id', 'item_tag_ids', 'count']].drop_duplicates(
            subset=['user_id', 'item_tag_ids']
        )

        df = df.merge(
            df.groupby(['user_id', 'item_tag_ids']).item_timestamp.agg(list).reset_index(),
            on=['user_id', 'item_tag_ids'],
            how='left',
            suffixes=['_1', '']
        ).drop('item_timestamp_1', axis=1)

        df2 = df.groupby(
            ['user_id', 'item_tag_ids']
        ).item_timestamp.agg(list).reset_index()

        listjoin = lambda x: [j for i in x for j in i]
        df2['item_timestamp'] = df2['item_timestamp'].apply(listjoin)
        df2['item_timestamp'] = df2['item_timestamp'].apply(lambda x: x[0])
        df2 = df2.sort_values(by=['user_id', 'item_timestamp'])
        df3 = pd.merge(df2, df_weights, on=["user_id", "item_tag_ids"], how="left")        
        cols = list(df3.columns)
        a, b = cols.index('item_timestamp'), cols.index('count')
        cols[b], cols[a] = cols[a], cols[b]
        df2 = df3[cols]

        # weights = np.array(df2['count'].values)
        # normalise = lambda v: v / np.sqrt(np.sum(v**2))
        # normalised_weights = normalise(weights)
        # df2['count'] = normalised_weights
        df2 = df2.rename(columns={'item_tag_ids':'item_id', 'count':'mentions'})
        df2 = df2.drop(['item_timestamp'], axis=1)
        df2['user_id'], df2['item_id'] = df2['user_id'].apply(str), df2['item_id'].apply(str)
        all_items = df2.groupby('user_id',)['item_id'].apply(list)
        all_counts = df2.groupby('user_id')['mentions'].apply(list)
        df2['all_item_id'] = dict
        df2['last_item'] = ''

        last_user_id, last_item_id = None, None
        for i, row in df2.iterrows():
            user_id = row['user_id']
            df2.at[i, 'all_item_id'] = dict(zip(['all_item_id_{0}'.format(i) for i in all_items[user_id]], all_counts[user_id]))
            if user_id != last_user_id:
                last_user_id, last_item_id = user_id, row['item_id']
                continue
            df2.at[i, 'last_item'], last_item_id = 'last_item_id_'+last_item_id, row['item_id']
        x, y = df2[['user_id', 'item_id', 'all_item_id', 'last_item']].copy(), df2['mentions'].copy()
        return (x, y)

    def df_to_libsvm(self, x: pd.DataFrame, y: pd.DataFrame):
        all_items, last_item = x.pop('all_item_id'), pd.get_dummies(x.pop('last_item'))
        x = pd.get_dummies(x)
        df_all_item_id = all_items.apply(pd.Series)
        df_all_item_id.fillna(value=0, inplace=True)
        df_all_item_id = df_all_item_id.div(df_all_item_id.sum(axis=1), axis=0)
        df_all_item_id = df_all_item_id.astype(np.float32)
        x = x.join(df_all_item_id, how='outer')
        x = x.join(last_item, how='outer')
        mat = x.as_matrix()
        dump_svmlight_file(mat, y, join(self._modelpath, 'svm-output.libsvm'))

    def run(self):
        x, y = self.csv_to_df(months=3)
        self.df_to_libsvm(x, y)
        

if __name__ == "__main__":
    # ab = AttributeParser('2017_04_01') # 2017_02_01
    # ab.run()
    # ctp = CashtagParser()
    # ctp.conversion_to_cashtag_orientation()
    dfm = xDeepFMModel()
    dfm.run()