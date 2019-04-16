from datetime import datetime

import pandas as pd

import os
import logging
import sys
import json
import re
import time

class DataPrep:
    def __init__(self, name):
        self.st_path = '/media/ntfs/st_2017'
        self.dpath = './data/csv/dataparser'
        self.logpath = './log/io/csv/dataparser'
        self.name = name

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/
        Also sets the formatting instructions for the log file, prints Time, Current Thread,
        Logging Type, Message.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1}_log ({2}).log".format(self.logpath, str(datetime.now())[:-7], self.name)),
                logging.StreamHandler(sys.stdout)
            ])
    
    def clear_logger_settings(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

class DataParser(DataPrep):
    def __init__(self, limit: str=None):
        super().__init__('data_parser')

        self.files = [f for f in os.listdir(self.st_path) if os.path.isfile(os.path.join(self.st_path, f))]
        self.files.sort()
        if limit:
            self.files = self.files[:next(self.files.index(x) for x in self.files if x.endswith(limit))]

    def get_symbols_only(self):
        logger = logging.getLogger()
        rows = []

        for i, fp in enumerate(self.files):
            logger.info("Parsing file <{}>, {}/{}".format(fp, i+1, len(self.files)))
            with open(os.path.join(self.st_path, fp)) as f:
                for line in f:
                    data = json.loads(line)['data']
                    symbols = data.get('symbols')
                    if not symbols:
                        continue

                    for s in symbols:
                        check = (s_id, s_industry, s_sector) = s.get('id'), s.get('industry'), s.get('sector')
                        if not all(check):
                            continue


                        s_industry = re.sub('[^\w]+', '', s_industry)
                        s_sector = re.sub('[^\w]+', '', s_sector)

                        timestamp = data.get('created_at')
                        timestamp = re.sub('T|Z', ' ', timestamp).strip()
                        timestamp = int(time.mktime(time.strptime(timestamp,'%Y-%m-%d %H:%M:%S')))
                        
                        rows.append({
                            'user_id': data['user']['id'],
                            'tag_id': s_id,
                            'timestamp': timestamp,
                            'tag_industry': s_industry,
                            'tag_sector': s_sector
                        })

        df = pd.DataFrame(rows, columns=['user_id', 'tag_id', 'timestamp','tag_industry', 'tag_sector'])
        
        outpath = os.path.join(self.dpath, '01_symbols.csv')
        df.to_csv(outpath, sep='\t', index=False)
            
    def run(self):
        self.logger()
        self.get_symbols_only()
        self.clear_logger_settings()

class DataCleaner(DataPrep):
    def __init__(self, fp):
        super().__init__('data_cleaner')
        self.df = pd.read_csv(os.path.join(self.dpath, fp), sep='\t')

    def interaction_threshold(self, k):
        logger = logging.getLogger()

        data_count = self.df.shape[0]
        logger.info("Begin cleaning... Dataframe Size: {}".format(data_count))
        self.df = self.df.groupby(['user_id', 'tag_id']).filter(lambda x: len(x) > k-1)
        logger.info("Removed users with less than {0} unique interactions. Size of DataFrame: {1} -> {2}".format(k, data_count, self.df.shape[0]))
        
        outpath = os.path.join(self.dpath, '02_interactions.csv')
        self.df.to_csv(outpath, sep='\t', index=False)

    def bot_cleaner(self):
        logger = logging.getLogger()
        bot_ids = [47688, 2843, 74023, 356080, 348830, 373849, 406225, 730219, 894342, 347689]

        data_count = self.df.shape[0]
        logger.info("Begin bot removal... Dataframe Size: {}".format(data_count))
        self.df = self.df[~self.df.user_id.isin(bot_ids)]
        logger.info("Removed bots. Size of DataFrame: {0} -> {1}".format(data_count, self.df.shape[0]))

        outpath = os.path.join(self.dpath, '03_bot_cleaned.csv')
        self.df.to_csv(outpath, sep='\t', index=False)
    
    def run(self, k):
        self.logger()

        self.interaction_threshold(k)
        self.bot_cleaner()
       
        self.clear_logger_settings()

dp = DataParser('2017_04_01')
dp.run()
dc = DataCleaner('01_symbols.csv')
# dc = DataCleaner('02_interactions.csv')
dc.run(50)

# da = DataAnalyser()
# da.run()