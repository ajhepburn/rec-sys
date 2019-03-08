import logging
import sys

from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import precision_recall_score, mrr_score

NUM_SAMPLES = 100

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [8, 16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]

class SpotlightImplicitModel:
    def __init__(self):
        self.logpath = './log/models/spotlightimplicitmodel/'
        self.rpath = './data/csv/cashtags_clean.csv'

    def logger(self):
        """Sets logger config to both std.out and to log ./log/io/csv/cleaner

        Also sets the formatting instructions for the log file, prints Time,
        Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1} ({2}).log".format(
                    self.logpath,
                    str(datetime.now())[:-7],
                    'spotlight_implicit_model',
                )),
                logging.StreamHandler(sys.stdout)
            ])

    def csv_to_df(self, months: int) -> tuple:
        """Reads in CSV file, converts it to a number of Pandas DataFrames.

        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and
                interactions between items.

        """

        df = pd.read_csv(self.rpath, sep='\t')
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
        df_interactions, df_timestamps = df[['user_id', 'item_tag_ids']], df['item_timestamp']
        return (df_interactions, df_timestamps, df_weights)

    def build_interactions_object(self, df_interactions: pd.DataFrame, df_timestamps: pd.DataFrame, df_weights: pd.DataFrame):
        logger = logging.getLogger()
        user_ids, cashtag_ids = df_interactions['user_id'].values.astype(int), df_interactions['item_tag_ids'].values.astype(int)
        timestamps, weights = df_timestamps.values, np.array(df_weights['count'].values)
        normalise = lambda v: v / np.sqrt(np.sum(v**2))
        normalised_weights = normalise(weights)
        interactions = Interactions(
            user_ids=user_ids, 
            item_ids=cashtag_ids, 
            timestamps=np.array([int(x[len(x)-1]) for x in timestamps]), 
            weights=normalised_weights
        )
        logger.info("Build interactions object: {}".format(interactions))
        return interactions

    def cross_validation(self, interactions: Interactions) -> tuple:
        logger = logging.getLogger()
        train, test = random_train_test_split(interactions)
        logger.info('Split into \n {} and \n {}.'.format(train, test))
        return (train, test)

    def sample_implicit_hyperparameters(self):
        pass

    def fit_implicit_model(self, train: Interactions) -> ImplicitFactorizationModel:
        logger = logging.getLogger()
        logger.info("Begin fitting implicit model...\n Hyperparameters: {0}".format('FILLER TEXT'))
        implicit_model = ImplicitFactorizationModel(params)
        implicit_model.fit(train, verbose=True)
        return implicit_model

    def evaluation(self, model, sets: tuple):
        logger = logging.getLogger()
        train, test = sets
        random_state = np.random.RandomState(100)

        logger.info("Beginning model evaluation...")

        train_mrr = mrr_score(model, train).mean()
        test_mrr = mrr_score(model, test).mean()
        logger.info('Train MRR {:.8f}, test MRR {:.8f}'.format(
            train_mrr, test_mrr
        ))


        train_prec, train_rec = precision_recall_score(model, train)
        test_prec, test_rec = precision_recall_score(model, test)
        logger.info('Train Precision@10 {:.8f}, test Precision@10 {:.8f}'.format(
            train_prec.mean(),
            test_prec.mean()
        ))
        logger.info('Train Recall@10 {:.8f}, test Recall@10 {:.8f}'.format(
            train_rec.mean(),
            test_rec.mean()
        ))

    def run(self):
        self.logger()
        df_interactions, df_timestamps, df_weights = self.csv_to_df(months=3)

        interactions = self.build_interactions_object(df_interactions, df_timestamps, df_weights)
        train, test = self.cross_validation(interactions)

        implicit_model = self.fit_implicit_model(train)

        self.evaluation(implicit_model, (train, test))

if __name__ == "__main__":
    sim = SpotlightImplicitModel()
    sim.run()
