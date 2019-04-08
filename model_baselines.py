from datetime import datetime

from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
from collections import Counter
import numpy as np
import implicit

from implicit.evaluation import train_test_split, mean_average_precision_at_k

import logging
import sys
import random
import time
import os

import pandas as pd

class BaselineModels:
    def __init__(self):
        self.logpath = './log/models/'
        self.rpath = './data/csv/cashtags_clean.csv'
        self.mpath = './models/'
        self.df = self.csv_to_df()

    def logger(self, model_name):
        """Sets the logger configuration to report to both std.out and to log to ./log/models/<MODEL_NAME>/
        
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} ({2}).log".format(self.logpath+'/baseline/'+model_name+'/', str(datetime.now())[:-7], model_name)),
                    logging.StreamHandler(sys.stdout)
                ])

    def csv_to_df(self) -> tuple:
        """Reads in CSV file declared in __init__ (self.rpath) and converts it to a number of Pandas DataFrames.
            
        Returns:
            tuple: Returns tuple of Pandas DataFrames; user features, item features and 
                interactions between items.

        """

        df = pd.read_csv(self.rpath, sep='\t')
        return df

class LightFMLib(BaselineModels):
    def __init__(self):
        super().__init__()

    def build_id_mappings(self, df_interactions: pd.DataFrame, df_user_features: pd.DataFrame, df_item_features: pd.DataFrame) -> Dataset:
        """Builds internal indice mapping for user-item interactions and encodes item features.

        Reads in user-item interactions and the features associated with each item and builds a mapping 
        between the user and item ids from our input data to indices that will be used internally by our model. 

        Item features are further encoded as an argument passed to Dataset.fit. These are supplied as a flat
        list of unique item features for the entire dataset.

        Args:
            df_interactions (pandas.DataFrame): User-Item interactions DataFrame consisting of user and item IDs.
            df_item_features (pandas.DataFrame): Item IDs and their corresponding features as column separated values.

        Returns:
            lightfm.data.Dataset: Tool for building interaction and feature matrices, 
                taking care of the mapping between user/item ids and feature names and internal feature indices.
            item_sectors (list): list of all the unique cashtag sector information in the dataset.
            item_industries (list): list of all the unique cashtag industries information in the dataset.
            item_cashtags (list): list of all the unique cashtag information in the dataset.

        """

        user_locations = list(map('LOC:{0}'.format, list(set('|'.join(df_user_features['user_location'].tolist()).split('|')))))
        item_sectors = list(map('SECTOR:{0}'.format, list(set('|'.join(df_item_features['item_sectors'].tolist()).split('|')))))
        item_industries = list(map('INDUSTRY:{0}'.format, list(set('|'.join(df_item_features['item_industries'].tolist()).split('|')))))
        item_cashtags = list(map('TAG:{0}'.format, list(set('|'.join(df_item_features['item_cashtags'].tolist()).split('|')))))
        item_timestamp = list(map('TIME:{0}'.format, list(df_item_features['item_timestamp'])))

        user_features = user_locations
        item_features = item_timestamp+item_sectors+item_industries+item_cashtags
        
        dataset = Dataset()
        dataset.fit((x for x in df_interactions['user_id']), 
                    (x for x in df_interactions['item_id']),
                    user_features=user_features,
                    item_features=item_features)
        return (dataset, user_locations, item_sectors, item_industries, item_cashtags)

    def build_interactions_matrix(self, dataset: Dataset, df_interactions: pd.DataFrame) -> tuple:
        """Builds a matrix of interactions between user and item.

        Takes as params a lightfm.data.Dataset object consisting of mapping between users
        and items and builds a matrix of interactions.

        Args:
            dataset (lightfm.data.Dataset): Dataset object consisting of internal user-item mappings.
            df_interactions (pandas.DataFrame): User-Item interactions DataFrame consisting of user and item IDs.

        Returns:
            tuple: Returns tuple with two scipy.sparse.coo_matrix matrices: the interactions matrix and the corresponding weights matrix.

        """

        def gen_rows(df):
            """Yields 

            Args:
               df (pd.DataFrame): df_interactions matrix

            Yields:
                pandas.DataFrame: User-Item interactions DataFrame consisting of user and item IDs

            Examples:
                Generates a row, line by line of user and item IDs to pass to the lightfm.data.Dataset.build_interactions function.

                >>> print(row)
                Pandas(user_id=123456, item_id=12345678)

            """
            for row in df.itertuples(index=False):
                yield row
                
        (interactions, weights) = dataset.build_interactions(gen_rows(df_interactions))
        return (interactions, weights)

    def build_user_features(self, dataset: Dataset, df_user_features: pd.DataFrame) -> csr_matrix:
        def gen_rows(df):
            """Yields 

            Args:
               df (pandas.DataFrame): df_user_features matrix

            Yields:
                pandas.DataFrame: User IDs and their corresponding features as column separated values.

            Examples:
                Generates a row, line by line of item IDs and their corresponding features/weights to pass to the 
                lightfm.data.Dataset.build_item_features function. The build_item_features function then normalises
                these weights per row.

                Also prepends each item with its type for a more accurate model.

                >>> print(row)
                [12345678, {'TAG:[CASHTAG]:2}]

            """

            for row in df.itertuples(index=False):
                d = row._asdict()
                user_locations = list(map('LOC:{0}'.format, d['user_location'].split('|') if '|' in d['user_location'] else [d['user_location']]))
                yield [d['user_id'], user_locations]

        user_features = dataset.build_user_features(gen_rows(df_user_features))
        return user_features

    def build_item_features(self, dataset: Dataset, df_item_features: pd.DataFrame) -> csr_matrix:
        """Binds item features to item IDs, provided they exist in the fitted model.

        Takes as params a lightfm.data.Dataset object consisting of mapping between users
        and items and a pd.DataFrame object of the item IDs and their corresponding features.

        Args:
            dataset (lightfm.data.Dataset): Dataset object consisting of internal user-item mappings.
            df_item_features (pandas.DataFrame): Item IDs and their corresponding features as column separated values.

        Returns:
            scipy.sparse.csr_matrix (num items, num features): Matrix of item features.

        """

        def gen_rows(df):
            """Yields 

            Args:
               df (pandas.DataFrame): df_item_features matrix

            Yields:
                pandas.DataFrame: Item IDs and their corresponding features as column separated values.

            Examples:
                Generates a row, line by line of item IDs and their corresponding features/weights to pass to the 
                lightfm.data.Dataset.build_item_features function. The build_item_features function then normalises
                these weights per row.

                Also prepends each item with its type for a more accurate model.

                >>> print(row)
                [12345678, {'TAG:[CASHTAG]:2}]

            """

            for row in df.itertuples(index=False):
                d = row._asdict()
                item_timestamp = 'TIME:'+str(d['item_timestamp'])
                item_sectors = list(map('SECTOR:{0}'.format, d['item_sectors'].split('|') if '|' in d['item_sectors'] else [d['item_sectors']]))
                item_industries =  list(map('INDUSTRY:{0}'.format, d['item_industries'].split('|') if '|' in d['item_industries'] else [d['item_industries']]))
                item_cashtags =  list(map('TAG:{0}'.format, d['item_cashtags'].split('|') if '|' in d['item_cashtags'] else [d['item_cashtags']]))

                weights_t = ({item_timestamp:1}, Counter(item_sectors), Counter(item_industries), Counter(item_cashtags))
                for weights_obj in weights_t:
                    for k, v in weights_obj.items():
                        yield [d['item_id'], {k:v}]


                # item_feature_list = item_sectors+item_industries+item_cashtags
                # yield [d['item_id'], item_feature_list]
        
        item_features = dataset.build_item_features(gen_rows(df_item_features), normalize=True)
        return item_features

    def cross_validate_interactions(self, interactions: coo_matrix) -> tuple:
        """Randomly split interactions between training and testing.

        This function takes an interaction set and splits it into two disjoint sets, a training set and a test set. 

        Args:
            interactions (scipy.sparse.coo_matrix): Matrix of user-item interactions.

        Returns:
            tuple: (scipy.sparse.coo_matrix, scipy.sparse.coo_matrix), A tuple of (train data, test data).

        """

        train, test = random_train_test_split(interactions)
        return train, test

    def evaluate_model(self, model: LightFM, model_name: str, eval_metrics: list, sets: tuple, NUM_THREADS: str, user_features: csr_matrix=None, item_features: csr_matrix=None, k: int=None):
        """Evaluates models on a number of metrics

        Takes model and evaluates it depending on which evaluation metrics are passed in.
        Has local functions auc, precrec and mrr corresponding to AUC ROC score, Precision@K/Recall@K, 
        Mean Reciprocal Rank metrics.

        Args:
            model (lightfm.LightFM): A LightFM model.
            model_name (str): The type of model being trained, for evaluation output purposes (Collaborative Filtering/Hybrid).
            eval_metrics (list): A list containing which evaluation metrics to carry out. Can be either of 'auc', 'precrec', 'mrr'
            sets (tuple): (scipy.sparse.coo_matrix, scipy.sparse.coo_matrix), A tuple of (train data, test data).
            NUM_THREADS (str): Number of threads to run evaluations on, corresponding to physical cores on system.
            user_features (scipy.sparse.csr_matrix, optional): Matrix of user features. Defaults to None.
            item_features (scipy.sparse.csr_matrix, optional): Matrix of item features. Defaults to None.
            k (integer, optional): The k parameter for Precision@K/Recall@K corresponding to Top-N recommendations.

        """

        logger = logging.getLogger()
        train, test = sets
        model_name = 'Collaborative Filtering' if model_name == 'cf' else 'Hybrid'

        def auc():
            """Evaluates models on the ROC AUC metric.

            Measure the ROC AUC metric for a model: the probability that a randomly chosen positive example 
            has a higher score than a randomly chosen negative example. A perfect score is 1.0.

            """

            train_auc = auc_score(model,
                            train,
                            user_features=user_features if user_features is not None else None,
                            item_features=item_features if item_features is not None else None,
                            num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set AUC: %s' % train_auc)

            test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    user_features=user_features if user_features is not None else None,
                    item_features=item_features if item_features is not None else None,
                    num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set AUC: %s' % test_auc)

        def precrec():
            """Evaluates models on Precision@K/Recall@K and also outputs F1 Score.

            Measure the precision at k metric for a model: the fraction of known positives in the first k 
            positions of the ranked list of results. A perfect score is 1.0.

            Measure the recall at k metric for a model: the number of positive items in the first k 
            positions of the ranked list of results divided by the number of positive items in the test period. #
            A perfect score is 1.0.

            Compute the F1 score, also known as balanced F-score or F-measure: The F1 score can be interpreted as a weighted 
            average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. 
            The relative contribution of precision and recall to the F1 score are equal.

            """

            train_precision = precision_at_k(model, 
                                train, 
                                k=k, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set Precision@%s: %s' % (k, train_precision))
            test_precision = precision_at_k(model, 
                                test, 
                                k=k, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set Precision@%s: %s' % (k, test_precision))

            train_recall = recall_at_k(model, 
                                train, 
                                k=k, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set Recall@%s: %s' % (k, train_recall))
            test_recall = recall_at_k(model, 
                                test, 
                                k=k, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set Recall@%s: %s' % (k, test_recall))

            f1_train, f1_test = 2*(train_recall * train_precision) / (train_recall + train_precision), 2*(test_recall * test_precision) / (test_recall + test_precision)
            logger.info(model_name+' training set F1 Score: %s' % (f1_train))
            logger.info(model_name+' test set F1 Score: %s' % (f1_test))

        def mrr():
            """Evaluates models on their Mean Reciprocal Rank.

            Measure the reciprocal rank metric for a model: 1 / the rank of the highest ranked positive example. 
            A perfect score is 1.0.

            """

            train_mrr = reciprocal_rank(model, 
                                train, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set MRR: %s' % (train_mrr))
            test_mrr = reciprocal_rank(model, 
                                test, 
                                user_features=user_features if user_features is not None else None,
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set MRR: %s' % (test_mrr))

        for metric in eval_metrics:
            locals()[metric]()

    def run(self):
        pass

class Implicit(BaselineModels):
    def __init__(self, model_type):
        super().__init__()
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        self.df = self.df[['user_id', 'item_id']]
        self.df = self.df.groupby(['user_id', 'item_id']).size().reset_index(name='weight')
        a = self.df.groupby('user_id')['weight'].transform('sum')
        self.df['weight'] = self.df['weight'].div(a)

        users = list(np.sort(self.df['user_id'].unique())) 
        items = list(self.df['item_id'].unique())
        weights = list(self.df['weight']) 

        rows = self.df['user_id'].astype('category', categories = users).cat.codes
        cols = self.df['item_id'].astype('category', categories = items).cat.codes 

        self.sparse_repr = csr_matrix((weights, (rows, cols)), shape=(len(users), len(items)))
        self.model_type = model_type

    def fit(self):
        model = implicit.als.AlternatingLeastSquares(factors=50) if self.model_type is 'als' else implicit.approximate_als.NMSLibAlternatingLeastSquares()
        model.fit(self.sparse_repr)
        return model

    def evaluate(self, model):
        train, test = train_test_split(self.sparse_repr, 0.8)
        mapk = mean_average_precision_at_k(model, train, test)
        print("MAP@10: {}".format(mapk))

    def run(self):
        model = self.fit()
        self.evaluate(model)

class LFMCF(LightFMLib):
    def __init__(self, loss):
        super().__init__()
        self.model_name = loss+'_cf'

    def cf_model(self, train: coo_matrix, params: tuple) -> LightFM:
        """Trains a pure collaborative filtering model.

        Args:
            train (scipy.sparse.coo_matrix): Training set as a COO matrix.
            params (tuple): A number of hyperparameters for the model, namely NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA.

        Returns:
            lightfm.LightFM: A lightFM model.

        """

        logger = logging.getLogger()
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

        # Run 3 epochs and time it.
        logger.info('Begin fitting collaborative filtering model...')
        model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
        return model

    def run(self):
        self.logger(self.model_name)
        logger = logging.getLogger()
        params = (NUM_THREADS, _, _, _) = (4,30,3,1e-16)

        df = self.csv_to_df()
        df_user_features, df_item_features, df_interactions = df[['user_id', 'user_location']], df[['item_id', 'item_timestamp', 'item_body','item_titles', 'item_cashtags', 'item_industries', 'item_sectors']], df[['user_id', 'item_id']]

        dataset, user_locations, item_sectors, item_industries, item_cashtags = self.build_id_mappings(df_interactions, df_user_features, df_item_features)
        interactions, _ = self.build_interactions_matrix(dataset, df_interactions)
        train, test = self.cross_validate_interactions(interactions)

        logger.info('The dataset has %s users and %s items with %s interactions in the test and %s interactions in the training set.' % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        cf_model = self.cf_model(train, params)
        self.evaluate_model(model=cf_model, model_name='cf', eval_metrics=['auc'], sets=(train, test), NUM_THREADS=NUM_THREADS)

        logger.info('There are {0} distinct user locations, {1} distinct sectors, {2} distinct industries and {3} distinct cashtags.'.format(len(user_locations), len(item_sectors), len(item_industries), len(item_cashtags)))

class LFMHybrid(LightFMLib):
    def __init__(self, loss):
        super().__init__()
        self.model_name = loss+'_hybrid'

    def hybrid_model(self, params: tuple, train: coo_matrix, user_features: csr_matrix=None, item_features: csr_matrix=None) -> LightFM:
        """Trains a hybrid collaborative filtering/content model

        Adds user/item features to model to enrich training data.

        Args:
            params (tuple): A number of hyperparameters for the model, namely NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA.
            train (scipy.sparse.coo_matrix): Training set as a COO matrix.
            user_features (scipy.sparse.csr_matrix) : Matrix of userfeatures.
            item_features (scipy.sparse.csr_matrix) : Matrix of item features.

        Returns:
            lightfm.LightFM: A lightFM model.

        """

        logger = logging.getLogger()
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        # Define a new model instance
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                        no_components=NUM_COMPONENTS)

        # Fit the hybrid model. Note that this time, we pass
        # in the item features matrix.
        logger.info('Begin fitting hybrid model...')
        model = model.fit(train,
                        item_features=item_features,
                        user_features=user_features,
                        epochs=NUM_EPOCHS,
                        num_threads=NUM_THREADS)
        return model

    def run(self):
        self.logger(self.model_name)
        logger = logging.getLogger()
        params = (NUM_THREADS, _, _, _) = (4,30,3,1e-16)

        df = self.csv_to_df()
        df_user_features, df_item_features, df_interactions = df[['user_id', 'user_location']], df[['item_id', 'item_timestamp', 'item_body','item_titles', 'item_cashtags', 'item_industries', 'item_sectors']], df[['user_id', 'item_id']]
        
        dataset, user_locations, item_sectors, item_industries, item_cashtags = self.build_id_mappings(df_interactions, df_user_features, df_item_features)
        interactions, _ = self.build_interactions_matrix(dataset, df_interactions)
        user_features = self.build_user_features(dataset, df_user_features)
        item_features = self.build_item_features(dataset, df_item_features)
        train, test = self.cross_validate_interactions(interactions)

        logger.info('The dataset has %s users and %s items with %s interactions in the test and %s interactions in the training set.' % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        hybrid_model = self.hybrid_model(params, train, user_features, item_features)
        self.evaluate_model(model=hybrid_model, model_name='h', eval_metrics=['auc', 'precrec', 'mrr'], sets=(train, test), NUM_THREADS=NUM_THREADS, user_features=user_features, item_features=item_features, k=10)

imp = Implicit('als')
imp.run()