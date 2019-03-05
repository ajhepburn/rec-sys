import os, json, sys, logging, csv
from datetime import datetime
import pandas as pd
import numpy as np

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
        titles, cashtags, trending_scores, watchlist_counts, exchanges, industries, sectors = [], [], [], [], [], [], []

        if symbols:                     # Checks to see if cashtags exist in tweet
            for s in symbols:
                check = (s_title, s_symbol, s_trending_score, s_watchlist_count, s_exchange, s_industry, s_sector) = s.get('title'), s.get('symbol'), s.get('trending_score'), s.get('watchlist_count'), s.get('exchange'), s.get('industry'), s.get('sector')  # Check to see if a cashtag contains an 'Industry' tag, otherwise skip
                if not all(check):
                    continue

                user_id, user_location = d.get('user').get('id'), d.get('user').get('location')
                item_id, item_timestamp, item_body = d.get('id'), d.get('created_at'), d.get('body').replace('\t',' ').replace('\n','')

                titles[len(titles):], cashtags[len(cashtags):], trending_scores[len(trending_scores):], watchlist_counts[len(watchlist_counts):], exchanges[len(exchanges):], industries[len(industries):], sectors[len(sectors):] = tuple(zip((s_title, s_symbol, s_trending_score, s_watchlist_count, s_exchange, s_industry, s_sector)))

        return (user_id, user_location, item_id, item_timestamp, item_body, titles, cashtags, trending_scores, watchlist_counts, exchanges, industries, sectors) if industries else ((None),)*12
                 
    def file_writer(self) -> int:
        """ Responsible for writing to ct_industry.csv in ./data/csv/ and logging each file read.
        
        Passes single line into self.parse which returns a tuple of metadata, this is then written
        to a single row in the CSV, provided the tuple returned by self.parse does not contain any None values.

        Returns:
            int: Returns the number of documents in the file.

        """

        logger = logging.getLogger()

        with open (os.path.join(self.wpath, 'metadata.csv'), 'w', newline='') as stocktwits_csv:
            fields = ['user_id', 'item_id', 'user_location', 'item_timestamp', 'item_body','item_titles','item_cashtags','item_tag_trending_score','item_tag_watchlist_count','item_exchanges','item_industries','item_sectors']
            writer = csv.DictWriter(stocktwits_csv, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            line_count = 0

            for fp in self.files:
                logger.info('Reading file {0}'.format(fp))
                with open(os.path.join(self.rpath, fp)) as f:
                    for l in f:
                        if not all(self.parse(l)):
                            continue
                        user_id, user_location, item_id, item_timestamp, item_body, titles, cashtags, trend_scores, watch_counts, exchanges, industries, sectors = self.parse(l)
                        writer.writerow({'user_id':user_id, 'item_id':item_id, 'user_location':user_location, 'item_body':item_body,'item_timestamp':item_timestamp, 'item_titles':'|'.join(map(str, titles)),'item_cashtags':'|'.join(map(str, cashtags)),'item_tag_trending_score':'|'.join(map(str, trend_scores)), 'item_tag_watchlist_count':'|'.join(map(str, watch_counts)), 'item_exchanges':'|'.join(map(str, exchanges)),'item_industries':'|'.join(map(str, industries)),'item_sectors':'|'.join(map(str, sectors))})
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
        for el in ('item_titles','item_cashtags', 'item_tag_trending_score','item_tag_watchlist_count', 'item_industries', 'item_sectors', 'item_exchanges'):
            df[el] = df[el].apply(lambda x: x.split('|') if '|' in x else [x])
        df0 = explode(df, ['item_titles','item_cashtags','item_tag_trending_score','item_tag_watchlist_count','item_exchanges','item_industries', 'item_sectors'], fill_value='')
        
        df0['item_tag_trending_score'] = df0['item_tag_trending_score'].astype(float)
        df0['item_tag_trending_score'] = (df0['item_tag_trending_score'] - min(df0['item_tag_trending_score']))/(max(df0['item_tag_trending_score']) - min(df0['item_tag_trending_score']))

        df0.to_csv(self.wpath, sep='\t')

if __name__ == "__main__":
    # ab = AttributeParser('2017_02_01') # 2017_02_01
    # ab.run()
    ctp = CashtagParser()
    ctp.conversion_to_cashtag_orientation()