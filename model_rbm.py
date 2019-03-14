from reco_utils.recommender.rbm.rbm import RBM
from reco_utils.dataset.python_splitters import numpy_stratified_split
from reco_utils.dataset.sparse import AffinityMatrix
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

from datetime import datetime

import logging
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TOP_K = 10

COLUMNS = {
    'col_user': 'user_id',
    'col_item': 'item_id',
    'col_rating': 'weight',
    'col_timestamp': 'item_timestamp'
}

class RestrictedBoltzmann:
    def __init__(self):
        self._logpath = './log/models/rbm/'
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
                        'restricted_boltzmann',
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

        # weights = np.array(df['count'].values)
        # normalise = lambda v: v / np.sqrt(np.sum(v**2))
        # normalised_weights = normalise(weights)
        # df['count'] = normalised_weights
        df = df.rename(columns={'item_tag_ids':'item_id', 'count':'weight'})
        return df

    def stratified_split(self, df: pd.DataFrame) -> tuple:
        col_formatted = {i:COLUMNS[i] for i in COLUMNS if i != 'col_timestamp'}
        am = AffinityMatrix(DF = df, **col_formatted)
        X = am.gen_affinity_matrix()
        Xtr, Xtst = numpy_stratified_split(X)
        print('train matrix size', Xtr.shape)
        print('test matrix size', Xtst.shape)
        return (
            am,
            Xtr,
            Xtst
        )

    def train_rbm_model(self, sets: tuple) -> RBM:
        train, test = sets
        model = RBM(
            hidden_units= 600, 
            training_epoch = 30, 
            minibatch_size= 60, 
            keep_prob=0.9,
            with_metrics =True
        )
        train_time = model.fit(train, test)
        return (
            model,
            train_time
        )

    def evaluation(self, test: pd.DataFrame, all_predictions: pd.DataFrame, time_train: float, time_test: float):
        params = {
            'rating_true': test,
            'rating_pred': all_predictions, 
            'col_prediction': 'prediction',
            'relevancy_method': 'top_k',
            'k': TOP_K
        }
        col_formatted = {i:COLUMNS[i] for i in COLUMNS if i != 'col_timestamp'}
        eval_map = map_at_k(**{**params, **col_formatted})

        eval_ndcg = ndcg_at_k(**{**params, **col_formatted})

        eval_precision = precision_at_k(**{**params, **col_formatted})

        eval_recall = recall_at_k(**{**params, **col_formatted})

        
        print("MAP:\t%f" % eval_map,
              "NDCG:\t%f" % eval_ndcg,
              "Precision@K:\t%f" % eval_precision,
              "Recall@K:\t%f" % eval_recall,
              "Train Time (s):\t%f" % time_train,
              "Test Time (s):\t%f" % time_test, 
              sep='\n'
        )
        
    def run(self):
        df = self.csv_to_df(months=3)
        am, train, test = self.stratified_split(df)
        model, time_train = self.train_rbm_model((train, test))
        top_k, time_test =  model.recommend_k_items(test)
        top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
        test_df = am.map_back_sparse(test, kind = 'ratings')
        self.evaluation(test_df, top_k_df, time_train, time_test)

if __name__ == "__main__":
    rbm = RestrictedBoltzmann()
    rbm.run()