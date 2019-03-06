import torch

from datetime import datetime
import logging, sys
import pandas as pd
from spotlight.interactions import Interactions, SequenceInteractions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import precision_recall_score, mrr_score
from statistics import median
import numpy as np
import itertools

class SpotlightImplicitModel:
    def __init__(self):
        self.logpath = './log/models/spotlightimplicitmodel/'
        self.rpath = './data/csv/cashtags_clean.csv'
        pass

    def logger(self):
        """Sets the logger configuration to report to both std.out and to log to ./log/io/csv/cleaner
        
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} ({2}).log".format(self.logpath, str(datetime.now())[:-7],'spotlight_implicit_model')),
                    logging.StreamHandler(sys.stdout)
                ])

    def csv_to_df(self) -> tuple:
        """Reads in CSV file declared in __init__ (self.rpath) and converts it to a number of Pandas DataFrames.
            
        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and 
                interactions between items.

        """

        df = pd.read_csv(self.rpath, sep='\t')
        # First we create count column with transform
        df['count'] = df.groupby(['user_id', 'item_tag_ids']).user_id.transform('size')
        occurrences = df['count'].tolist()
        occ_cutoff = 2 * median(occurrences)
        
        df_weights = df[['user_id', 'item_tag_ids', 'count']].drop_duplicates(subset=['user_id', 'item_tag_ids'])
        df_weights = df_weights[df_weights['count'] < occ_cutoff]

        # AFter that we merge our groupby with apply list back to our original dataframe
        df = df.merge(df.groupby(['user_id', 'item_tag_ids']).item_timestamp.agg(list).reset_index(), 
                    on=['user_id', 'item_tag_ids'], 
                    how='left',
                        suffixes=['_1', '']).drop('item_timestamp_1', axis=1)
        df = df.groupby(['user_id', 'item_tag_ids']).item_timestamp.agg(list).reset_index()
        listjoin =  lambda x: [j for i in x for j in i]
        df['item_timestamp'] = df['item_timestamp'].apply(listjoin)
        df_interactions, df_timestamps = df[['user_id', 'item_tag_ids']], df['item_timestamp']
        return (df_interactions, df_timestamps, df_weights)

    def build_interactions_object(self, df_interactions: pd.DataFrame, df_timestamps: pd.DataFrame, df_weights: pd.DataFrame, seq: bool):
        logger = logging.getLogger()
        user_ids, cashtag_ids, timestamps, weights = df_interactions['user_id'].values.astype(int), df_interactions['item_tag_ids'].values.astype(int), df_timestamps.values, np.array(df_weights['count'].values)
        normalise = lambda v: v / np.sqrt(np.sum(v**2))
        normalised_weights = normalise(weights)
        interactions = Interactions(user_ids=user_ids, item_ids=cashtag_ids, timestamps=np.array([int(x[0]) for x in timestamps]), weights=normalised_weights)
        if seq:
            interactions.to_sequence()
        logger.info("Build interactions object: {}".format(interactions))
        return interactions

    def cross_validation(self, interactions: Interactions) -> tuple:
        logger = logging.getLogger()
        train, test = random_train_test_split(interactions)
        logger.info('Split into \n {} and \n {}.'.format(train, test))
        return (train, test)

    def fit_implicit_model(self, train: Interactions) -> ImplicitFactorizationModel:
        logger = logging.getLogger()
        logger.info("Begin fitting implicit model...")
        implicit_model = ImplicitFactorizationModel(use_cuda=True)
        implicit_model.fit(train, verbose=True)
        return implicit_model

    def fit_sequence_model(self, train: SequenceInteractions) -> ImplicitSequenceModel:
        model = ImplicitSequenceModel()
        return model

    def evaluation(self, model, sets: tuple):
        logger = logging.getLogger()
        train, test = sets

        logger.info("Beginning model evaluation...")
        
        train_mrr = mrr_score(model, train).mean()
        test_mrr = mrr_score(model, test).mean()
        logger.info('Train MRR {:.8f}, test MRR {:.8f}'.format(train_mrr, test_mrr))
    

        train_prec, train_rec = precision_recall_score(model, train)
        test_prec, test_rec = precision_recall_score(model, test)
        logger.info('Train Precision@10 {:.8f}, test Precision@10 {:.8f}'.format(train_prec.mean(), test_prec.mean()))
        logger.info('Train Recall@10 {:.8f}, test Recall@10 {:.8f}'.format(train_rec.mean(), test_rec.mean()))

    def run(self, seq):
        self.logger()
        df_interactions, df_timestamps, df_weights = self.csv_to_df()

        interactions = self.build_interactions_object(df_interactions, df_timestamps, df_weights, seq)
        train, test = self.cross_validation(interactions)
        
        implicit_model = self.fit_implicit_model(train)
        
        # SEQ_PARAMS = {
        #     'loss': ,
        #     'representation' ,
        #     'batch_size': ,
        #     'learning_rate': ,
        #     'l2': ,
        #     'n_iter': ,
        #     'random_state': ,
        # }
 
        self.evaluation(implicit_model, (train, test))

        
if __name__ == "__main__":
    sim = SpotlightImplicitModel()
    sim.run(seq=True)