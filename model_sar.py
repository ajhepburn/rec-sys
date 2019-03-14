from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

from datetime import datetime

import numpy as np
import pandas as pd
import logging
import sys

TOP_K = 10

class SmartAdaptiveRec:
    def __init__(self):
        self._logpath = './log/models/spotlightimplicitmodel/'
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

        df = df.groupby(
            ['user_id', 'item_tag_ids']
        ).item_timestamp.agg(list).reset_index()

        listjoin = lambda x: [j for i in x for j in i]
        df['item_timestamp'] = df['item_timestamp'].apply(listjoin)
        df['item_timestamp'] = df['item_timestamp'].apply(lambda x: x[0])
        df3 = pd.merge(df, df_weights, on=["user_id", "item_tag_ids"], how="left")        
        cols = list(df3.columns)
        a, b = cols.index('item_timestamp'), cols.index('count')
        cols[b], cols[a] = cols[a], cols[b]
        df = df3[cols]

        weights = np.array(df['count'].values)
        normalise = lambda v: v / np.sqrt(np.sum(v**2))
        normalised_weights = normalise(weights)
        df['count'] = normalised_weights
        df = df.rename(columns={'item_tag_ids':'item_id','count':'weight'})
        print(type(df['weight']))
        sys.exit(0)
        return df

    def run(self):
        df = self.csv_to_df(months=3)

if __name__ == "__main__":
    sar = SmartAdaptiveRec()
    sar.run()