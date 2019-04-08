import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
from datetime import datetime
from collections import Counter
import numpy as np

import logging, sys


class HybridBaselineCashtagRecommender:
    def __init__(self):
        self.logpath = './log/models/hybridbaselinecashtagmodel/'
        self.rpath = './data/csv/cashtags_clean.csv'
        self.mpath = './models/'

    def logger(self):
        """Sets the logger configuration to report to both std.out and to log to ./log/io/csv/cleaner
        
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} ({2}).log".format(self.logpath, str(datetime.now())[:-7],'hybrid_baseline_cashtag')),
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

    def build_id_mappings(self, df_interactions: pd.DataFrame, df_user_features: pd.DataFrame, df_item_features: pd.DataFrame) -> Dataset:
        user_locations = list(map('LOC:{0}'.format, list(set('|'.join(df_user_features['user_location'].tolist()).split('|')))))
        item_exchanges = list(map('EXCH:{0}'.format, list(set(df_item_features['item_exchanges'].tolist()))))
        item_sectors = list(map('SECTOR:{0}'.format, list(set(df_item_features['item_sectors'].tolist()))))
        item_industries = list(map('INDUSTRY:{0}'.format, list(set(df_item_features['item_industries'].tolist()))))
        # item_trending_scores = list(map('TS:{0}'.format, list(set(df_item_features['item_tag_trending_score'].tolist()))))
        # item_watchlist_counts = list(map('WC:{0}'.format, list(set(df_item_features['item_tag_watchlist_count'].tolist()))))

        user_features = user_locations
        item_features = list(set(df_item_features['item_tag_trending_score'].tolist()))+list(set(df_item_features['item_tag_watchlist_count'].tolist()))+item_exchanges+item_sectors+item_industries

        dataset = Dataset()
        dataset.fit((x for x in df_interactions['user_id']), 
                    (x for x in df_interactions['item_cashtags']),
                    user_features=user_features,
                    item_features=item_features)
        return dataset

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
        # df = df_user_features.groupby('user_id')['user_location'].apply('|'.join).reset_index()
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
                # loc_weights = Counter(user_locations)

                # for k, v in loc_weights.items():
                #     yield [d['user_id'], {k:v}]

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
                item_tag_trending_score, item_tag_watchlist_count, item_exchange, item_sector, item_industry = d['item_tag_trending_score'], d['item_tag_watchlist_count'], "EXCH:"+d['item_exchanges'], "SECTOR:"+d['item_sectors'], "INDUSTRY:"+d['item_industries']
                yield [d['item_cashtags'], [item_tag_trending_score, item_tag_watchlist_count, item_exchange, item_sector, item_industry]]

                # weights_t = ({item_timestamp:1}, Counter(item_sectors), Counter(item_industries), Counter(item_cashtags))
                # for weights_obj in weights_t:
                #     for k, v in weights_obj.items():
                #         yield [d['item_cashtags'], {k:v}]


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

    def cf_model_pure(self, train: coo_matrix, params: tuple) -> LightFM:
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

        logger.info('Begin fitting collaborative filtering model...')
        model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
        return model

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
        self.logger()
        logger = logging.getLogger()
        params = (NUM_THREADS, _, _, _) = (4,30,3,1e-16)

        data = self.csv_to_df()
        df_interactions = data[['user_id', 'item_cashtags']]
        df_interactions = df_interactions.groupby(['user_id','item_cashtags']).size().reset_index() \
                                               .rename(columns={0:'interactions'})
        df_item_features = data[['item_titles','item_cashtags', 'item_tag_trending_score','item_tag_watchlist_count','item_exchanges','item_sectors', 'item_industries']].drop_duplicates()
        df_user_features = data[['user_id', 'user_location']].drop_duplicates()

        dataset = self.build_id_mappings(df_interactions, df_user_features, df_item_features)
        interactions, weights = self.build_interactions_matrix(dataset, df_interactions)

        user_features = self.build_user_features(dataset, df_user_features)
        item_features = self.build_item_features(dataset, df_item_features)

        train, test = self.cross_validate_interactions(interactions)
        cf_model = self.cf_model_pure(train, params)
        self.evaluate_model(model=cf_model, model_name='cf', eval_metrics=['auc', 'precrec', 'mrr'], sets=(train, test), NUM_THREADS=NUM_THREADS, k=10)

        hybrid_model = self.hybrid_model(params, train, user_features, item_features)
        self.evaluate_model(model=hybrid_model, model_name='h', eval_metrics=['auc', 'precrec', 'mrr'], sets=(train, test), NUM_THREADS=NUM_THREADS, user_features=user_features, item_features=item_features, k=10)
        

cashrec = HybridBaselineCashtagRecommender()
cashrec.run()