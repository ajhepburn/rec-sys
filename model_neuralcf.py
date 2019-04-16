from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items, auc)

from datetime import datetime

import logging
import sys
import time
import os

import pandas as pd
import numpy as np
import tensorflow as tf

# top k items to recommend
TOP_K = 5

# Model parameters
EPOCHS = 50
BATCH_SIZE = 256

COLUMNS = {
    'col_user': 'user_id',
    'col_item': 'item_id',
    'col_rating': 'weight',
    'col_timestamp': 'timestamp'
}


class NeuralCFModel:
    def __init__(self):
        self._logpath = './log/models/neuralcf/'
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
                        'neural_cf',
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
        # df = df[df['count'] < months*100]
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


        # weights = np.array(df['count'].values)
        # normalise = lambda v: v / np.sqrt(np.sum(v**2))
        # normalised_weights = normalise(weights)
        df['count'] = df.groupby('user_id')['count'].transform(lambda x: x/x.sum()) # THIS IS RIGHT
        df = df.rename(columns={'tag_id':'item_id', 'count':'weight'})
        return df

    def split_train_test(self, df: pd.DataFrame) -> NCFDataset:
        col_formatted = {i:COLUMNS[i] for i in COLUMNS if i != 'col_rating'}
        train, test = python_chrono_split(**{**{'data':df}, **col_formatted})

        ncf_params = {
            'train':train,
            'test':test,
            'seed':42
        }
        data = NCFDataset(**{**ncf_params, **COLUMNS})
        return (
            data,
            train,
            test
        )

    def fit_ncf_model(self, data: NCFDataset) -> NCF:
        model = NCF (
            n_users=data.n_users, 
            n_items=data.n_items,
            model_type="NeuMF",
            n_factors=4,
            layer_sizes=[16,8,4],
            n_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=1e-3,
            verbose=10,
        )

        start_time = time.time()

        model.fit(data)

        train_time = time.time() - start_time

        print("Took {} seconds for training.".format(train_time))
        return model

    def predict_model(self, model: NCF, train: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()

        users, items, preds = [], [], []
        item = list(train.item_id.unique())
        for user in train.user_id.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"user_id": users, "item_id":items, "prediction":preds})

        merged = pd.merge(train, all_predictions, on=["user_id", "item_id"], how="outer")
        all_predictions = merged[merged.weight.isnull()].drop('weight', axis=1)

        test_time = time.time() - start_time
        print("Took {} seconds for prediction.".format(test_time))
        return all_predictions

    def evaluation(self, test: pd.DataFrame, all_predictions: pd.DataFrame):
        params = {
            'rating_true': test,
            'rating_pred': all_predictions,
            'col_prediction': 'prediction',
            'k': TOP_K
        }
        auc_params = {i:params[i] for i in params if i!='k'}
        col_formatted = {i:COLUMNS[i] for i in COLUMNS if i != 'col_timestamp'}
        # eval_auc = auc(**{**auc_params, **col_formatted})
        eval_map = map_at_k(**{**params, **col_formatted})
        eval_ndcg = ndcg_at_k(**{**params, **col_formatted})
        eval_precision = precision_at_k(**{**params, **col_formatted})
        eval_recall = recall_at_k(**{**params, **col_formatted})

        print(
            "MAP:\t%f" % eval_map,
            "NDCG:\t%f" % eval_ndcg,
            "Precision@K:\t%f" % eval_precision,
            "Recall@K:\t%f" % eval_recall, sep='\n')

    def run(self):
        df = self.csv_to_df(months=6)
        data, train, test = self.split_train_test(df)
        model = self.fit_ncf_model(data)
        predictions = self.predict_model(model, train)
        self.evaluation(test, predictions)


if __name__ == "__main__":
    ncf = NeuralCFModel()
    ncf.run()