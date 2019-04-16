from reco_utils.dataset.python_splitters import python_random_split, python_stratified_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

from datetime import datetime

import numpy as np
import pandas as pd
import logging
import sys
import time

TOP_K = 5
COLUMNS = {
    'col_user': 'user_id',
    'col_item': 'item_id',
    'col_rating': 'weight',
    'col_timestamp': 'timestamp'
}

class SmartAdaptiveRec:
    def __init__(self):
        self._logpath = './log/models/smartadaptiverec/'
        self._rpath = './data/csv/dataparser/03_bot_cleaned.csv'

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

    def csv_to_df(self, months: int) -> pd.DataFrame:
        """Reads in CSV file, converts it to a number of Pandas DataFrames.

        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and
                interactions between items.

        """

        df = pd.read_csv(self._rpath, sep='\t')
        df['count'] = df.groupby(['user_id', 'tag_id']).user_id.transform('size')
        df = df[df['count'] < months*100]
        df_weights = df[['user_id', 'tag_id', 'count']].drop_duplicates(
            subset=['user_id', 'tag_id']
        )

        df = df.merge(
            df.groupby(['user_id', 'tag_id']).timestamp.agg(list).reset_index(),
            on=['user_id', 'tag_id'],
            how='left',
            suffixes=['_1', '']
        ).drop('timestamp_1', axis=1)

        df = df.groupby(
            ['user_id', 'tag_id']
        ).timestamp.agg(list).reset_index()

        listjoin = lambda x: [j for i in x for j in i]
        df['timestamp'] = df['timestamp'].apply(listjoin)
        df['timestamp'] = df['timestamp'].apply(lambda x: x[0])
        df3 = pd.merge(df, df_weights, on=["user_id", "tag_id"], how="left")        
        cols = list(df3.columns)
        a, b = cols.index('timestamp'), cols.index('count')
        cols[b], cols[a] = cols[a], cols[b]
        df = df3[cols]

        weights = np.array(df['count'].values)
        normalise = lambda v: v / np.sqrt(np.sum(v**2))
        normalised_weights = normalise(weights)
        df['count'] = normalised_weights
        df = df.rename(columns={'tag_id':'item_id', 'count':'weight'})
        return df

    def fit_sar_model(self, train: pd.DataFrame) -> SARSingleNode:
        sar_params = {
            'similarity_type':"jaccard", 
            'time_decay_coefficient':30, 
            'time_now':None, 
            'timedecay_formula':True,
        }
        model = SARSingleNode(**{**sar_params, **COLUMNS})
        
        start_time = time.time()

        model.fit(train)

        train_time = time.time() - start_time
        print("Took {} seconds for training.".format(train_time))
        return model

    def predict_items(self, model: SARSingleNode, test: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()

        top_k = model.recommend_k_items(test, remove_seen=True)

        test_time = time.time() - start_time
        print("Took {} seconds for prediction.".format(test_time))
        return top_k

    def evaluation(self, model: SARSingleNode, test: pd.DataFrame, top_k: pd.DataFrame):
        params = {
            'rating_true': test,
            'rating_pred': top_k,
            'k':TOP_K
        }
        col_formatted = {i:COLUMNS[i] for i in COLUMNS if i != 'col_timestamp'}
        eval_map = map_at_k(**{**params, **col_formatted})
        eval_ndcg = ndcg_at_k(**{**params, **col_formatted})
        eval_precision = precision_at_k(**{**params, **col_formatted})
        eval_recall = recall_at_k(**{**params, **col_formatted})
        print(
            "Model:\t" + model.model_str,
            "Top K:\t%d" % TOP_K,
            "MAP:\t%f" % eval_map,
            "NDCG:\t%f" % eval_ndcg,
            "Precision@K:\t%f" % eval_precision,
            "Recall@K:\t%f" % eval_recall, sep='\n'
        )

    def run(self):
        df = self.csv_to_df(months=3)
        # train, test = python_random_split(df)
        train, test = python_stratified_split(
            df, filter_by="user", ratio=0.7,
            col_user='user_id', col_item='item_id'
        )
        model = self.fit_sar_model(train)
        top_k = self.predict_items(model, test)
        self.evaluation(model, test, top_k)



if __name__ == "__main__":
    sar = SmartAdaptiveRec()
    sar.run()