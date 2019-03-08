"""Spotlight Model Trainer

This file is responsible for training a number of Spotlight models
on the StockTwits dataset provided. Contained within are a number of
models, namely:
                SpotlightImplicitModel
                SpotlightSequenceModel

"""

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
    """ Responsible for appending model evaluation/hyperparameter information.

    Attributes:
        _respath (str): The path for storing results files, usually held
            in the model's log folder.
        _filename (str): The path to the output results file.
        _hashlib (md5 hash): Hashes the hyperparameter results for easy parsing..

    """

    def __init__(self, filename: str):
        self._respath = './log/models/spotlightimplicitmodel/results/'
        self._filename = filename
        self._hash = lambda x: hashlib.md5(
            json.dumps(x, sort_keys=True).encode('utf-8')
        ).hexdigest()
        open(self._respath+self._filename, 'a+')

    def save(self, evaluation: dict, hyperparameters: dict):
        """Saves the output from a model evaluation instance to file.

        Args:
            evaluation (dict): Precision, Recall and Mean Reciprocal
                Rank metrics from a single model evaluation.
            hyperparameters (dict): Hyperparameters used by model.

        """

        result = {
            'hyperparameters': self._hash(hyperparameters),
            'evaluation': evaluation
        }
        with open(self._respath+self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):
        """Returns best performing evaluation results and their hyperparameters.

        Returns:
            dict: Returns best result from current results file.

        """

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


class SpotlightModel:
    """SpotlightModel runs the model training and output.

    This trainer works through its main ``run`` function, which
    calls other functions within the class.

    An input CSV is read in and split into appropriate Pandas DataFrame objects,
    extra functionality is included for performance and readability; a hyperparameter
    sampler and logging function, respectively.

    A Spotlight model must first have an interactions object, mapping entities to
    internal IDs. These are then split into training/test sets and fed into a number
    of different kinds of models, from matrix factorization to deep learning architectures.
    These are then evaluated by a number of metrics and their results are saved via a
    Results instance, defined above.

    Attributes:
        _logpath (str): The path for storing model log files.
        _rpath (str): The path to the input data.

    """

    def __init__(self):
        self._logpath = './log/models/spotlightimplicitmodel/'
        self._rpath = './data/csv/cashtags_clean.csv'

    def logger(self):
        """Sets logger config to both std.out and log ./log/io/csv/spotlightimplicitmodel/

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
        df_interactions, df_timestamps = df[['user_id', 'item_tag_ids']], df['item_timestamp']
        return (
            df_interactions,
            df_timestamps,
            df_weights
        )

    def build_interactions_object(self, df_interactions: pd.DataFrame, df_timestamps: pd.DataFrame, df_weights: pd.DataFrame) -> Interactions:
        """Builds a matrix of interactions between user and cashtag item.

        Takes as params a number of pandas.DataFrame which contains mappings between
        user-cashtag interactions, associated timestamps, normalised weights for
        interactions and builds a matrix for input to a Spotlight model.

        Args:
            df_interactions (pandas.DataFrame): User-Item interactions DataFrame consisting of user and item IDs.
            df_timestamps (pandas.DataFrame): Timestamps DataFrame consisting of timestamps associated with mappings
                in df_interactions.
            df_weights (pandas.DataFrame): Weights DataFrame consisting of weights associated with mappings
                in df_interactions, that is, the number of times a user has interacted with a particular item.

        Returns:
            spotlight.interactions.Interactions: Returns Spotlight interactions matrix.

        """

        logger = logging.getLogger()
        user_ids = df_interactions['user_id'].values.astype(int)
        cashtag_ids = df_interactions['item_tag_ids'].values.astype(int)
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
        """Randomly split interactions between training and testing.

        This function takes an interaction set and splits it into two disjoint sets,
        a training set and a test set.

        Args:
            interactions (spotlight.interactions.Interactions): Matrix of user-item interactions.

        Returns:
            tuple: (spotlight.interactions.Interactions, spotlight.interactions.Interactions),
                A tuple of (train data, test data).

        """

        logger = logging.getLogger()
        train, test = random_train_test_split(interactions)
        logger.info('Split into \n {} and \n {}.'.format(train, test))
        return (
            train,
            test
        )

    def sample_implicit_hyperparameters(self, random_state: np.random.RandomState, num: int) -> dict:
        """Randomly samples hyperparameters for input to model.

        Uses sklearn.model_selection.ParameterSampler to sample random hyperparameters
        from a given dictionary which are then yielded to a model.

        Args:
            random_state (np.random.RandomState): Random state to use when fitting.
            num (int): Number of samples to generate.

        Returns:
            dict: Returns dict of sampled hyperparameters.

        """

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

    def fit_implicit_model(self, train: Interactions, random_state: np.random.RandomState, hyperparameters: dict = None) -> ImplicitFactorizationModel:
        """Trains a Spotlight implicit matrix factorization model.

        Args:
            train (spotlight.interactions.Interactions): Training set as an interactions matrix.
            random_state (np.random.RandomState): Random state to use when fitting.
            hyperparameters (dict, optional): A number of hyperparameters for the model, either sampled
                from sample_implicit_hyperparameters or default used by model. Defaults can be found
                in global variable DEFAULT_PARAMS.

        Returns:
            spotlight.factorization.implicit.ImplicitFactorizationModel: A Spotlight implicit matrix factorization model.

        """

        logger = logging.getLogger()
        if hyperparameters:
            logger.info("Beginning fitting implicit model... \n Hyperparameters: \n {0}".format(
                json.dumps({i:hyperparameters[i] for i in hyperparameters if i != 'use_cuda'})
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
        """Evaluates models on a number of metrics

        Takes model and evaluates it by Precision@K/Recall@K, Mean Reciprocal Rank metrics.

        Args:
            model (Arbitrary): A Spotlight model, can be of different types.
            sets (tuple): (spotlight.interactions.Interactions, spotlight.interactions.Interactions), A tuple of (train data, test data).

        Returns:
            dict: A dictionary with all the evaluation metrics.

        """

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

    def run(self, results_file: str = None, default: str = False):
        """Main function of class which calls on other methods to run model.

        Args:
            results_file (str, optional): Can be passed in to append to existing
                results file.
            default (str, optional): If true, default hyperparameters will be run
                for each model.

        """

        self.logger()
        logger = logging.getLogger()
        random_state = np.random.RandomState(100)
        init_time = str(datetime.now())[:-7]

        if not results_file:
            results = Results('{}_results.txt'.format(init_time))
        else:
            results = Results('{0}results/{1}.txt'.format(self._logpath, results_file))
        best_result = results.best()

        df_interactions, df_timestamps, df_weights = self.csv_to_df(months=3)

        interactions = self.build_interactions_object(df_interactions, df_timestamps, df_weights)
        train, test = self.cross_validation(interactions)

        if not default:
            for hyperparameters in self.sample_implicit_hyperparameters(random_state, NUM_SAMPLES):
                if hyperparameters in results:
                    continue

                implicit_model = self.fit_implicit_model(
                    hyperparameters=hyperparameters,
                    train=train,
                    random_state=random_state
                )
                evaluation = self.evaluation(implicit_model, (train, test))
                results.save(evaluation, hyperparameters)
        else:
            implicit_model = self.fit_implicit_model(train=train, random_state=random_state)
            evaluation = self.evaluation(implicit_model, (train, test))
            results.save(DEFAULT_PARAMS, evaluation)

        if best_result is not None:
            logger.info('Best result: {}'.format(results.best()))

if __name__ == "__main__":
    sim = SpotlightModel()
    sim.run(default=True)
