import hashlib
import logging
import sys
import json

from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterSampler

from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import precision_recall_score, mrr_score

NUM_SAMPLES = 100
DEFAULT_PARAMS = {
            'learning_rate': 0.01,
            'loss': 'pointwise',
            'batch_size': 256,
            'embedding_dim': 32,
            'n_iter': 10,
            'l2': 0.0
        }

class Results:
    def __init__(self, filename: str):
        self._respath = './log/models/spotlightimplicitmodel/results/'
        self._filename = filename
        self._hash = lambda x : hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()
        open(self._respath+self._filename, 'a+')

    def save(self, evaluation: dict, hyperparameters: dict):
        result = {
            'hyperparameters': self._hash(hyperparameters),
            'evaluation': evaluation
        }
        with open(self._respath+self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):
        results = sorted([x for x in self],
                         key=lambda x: -x['test']['mrr'])

        if results:
            return results[0]
        return None

    def __getitem__(self, hyperparams):
        params_hash = self._hash(hyperparams)
        with open(self._respath+self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hyperparameters'] == params_hash:
                    del datum['hyperparameters']
                    return datum
        raise KeyError

    def __contains__(self, x):
        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with open(self._respath+self._filename, 'r+') as f:
            for line in f:
                datum = json.loads(line)
                del datum['hyperparameters']
                yield datum


class SpotlightImplicitModel:
    def __init__(self, default=False):
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
            timestamps=np.array([int(x[0]) for x in timestamps]), 
            weights=normalised_weights
        )
        logger.info("Build interactions object: {}".format(interactions))
        return interactions

    def cross_validation(self, interactions: Interactions) -> tuple:
        logger = logging.getLogger()
        train, test = random_train_test_split(interactions)
        logger.info('Split into \n {} and \n {}.'.format(train, test))
        return (train, test)

    def sample_implicit_hyperparameters(self, random_state: np.random.RandomState, num: int) -> dict:
        space = {
            'learning_rate': [1e-3, 1e-2, 5 * 1e-2, 1e-1],
            'loss': ['bpr', 'hinge', 'adaptive_hinge', 'pointwise'],
            'batch_size': [8, 16, 32, 256],
            'embedding_dim': [8, 16, 32, 64, 128, 256],
            'n_iter': list(range(5, 20)),
            'l2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]
        }

        sampler = ParameterSampler(
            space,
            n_iter=num,
            random_state=random_state
        )

        for params in sampler:
            yield params
        

    def fit_implicit_model(self, train: Interactions, random_state: np.random.RandomState, hyperparameters: dict=None) -> ImplicitFactorizationModel:
        logger = logging.getLogger()
        if hyperparameters:
            logger.info("Beginning fitting implicit model... \n Hyperparameters: \n {0}".format(
                json.dumps({i:hyperparameters[i] for i in hyperparameters if i!='use_cuda'})
            ))
            implicit_model = ImplicitFactorizationModel(
                loss=hyperparameters['loss'],
                learning_rate=hyperparameters['learning_rate'],
                batch_size=hyperparameters['batch_size'],
                embedding_dim=hyperparameters['embedding_dim'],
                n_iter=hyperparameters['n_iter'],
                l2=hyperparameters['l2'],
                use_cuda=True,
                random_state=random_state
            )
        else:
            logger.info("Beginning fitting implicit model with default hyperparameters...")
            implicit_model = ImplicitFactorizationModel(use_cuda=True)
        implicit_model.fit(train, verbose=True)
        return implicit_model

    def evaluation(self, model, sets: tuple):
        logger = logging.getLogger()
        train, test = sets

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
        return {
            'train': {
                'precision':train_prec.mean(),
                'recall':train_rec.mean(),
                'mrr':train_mrr,
            },
            'test': {
                'precision':test_prec.mean(),
                'recall':test_rec.mean(),
                'mrr':test_mrr,
            },
        }

    def run(self, results_file: str=None, default: str=False):
        self.logger()
        logger = logging.getLogger()
        random_state = np.random.RandomState(100)
        init_time = str(datetime.now())[:-7]

        if not results_file:
            results = Results('{}_results.txt'.format(init_time))
        else:
            results = Results('{0}results/{1}.txt'.format(self.logpath, results_file))
        best_result = results.best()

        df_interactions, df_timestamps, df_weights = self.csv_to_df(months=3)

        interactions = self.build_interactions_object(df_interactions, df_timestamps, df_weights)
        train, test = self.cross_validation(interactions)
        
        if not default:
            for hyperparameters in self.sample_implicit_hyperparameters(random_state, NUM_SAMPLES):
                if hyperparameters in results:
                    continue

                implicit_model = self.fit_implicit_model(hyperparameters=hyperparameters, train=train, random_state=random_state)
                evaluation = self.evaluation(implicit_model, (train, test))
                results.save(evaluation, hyperparameters)
        else:
            implicit_model = self.fit_implicit_model(train=train, random_state=random_state)
            evaluation = self.evaluation(implicit_model, (train, test))
            results.save(DEFAULT_PARAMS, evaluation)
        
        if best_result is not None:
            logger.info('Best result: {}'.format(results.best()))

if __name__ == "__main__":
    sim = SpotlightImplicitModel()
    sim.run(default=True)
