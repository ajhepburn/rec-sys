import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
from datetime import datetime

import sys, regex, logging

class UserCashtagBaselineModel:
    def __init__(self):
        self.logpath = './log/models/lfm_hybrid/'
        self.rpath = './data/csv/metadata_clean.csv'

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/cleaner
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1}_log ({2}).log".format(self.logpath, str(datetime.now())[:-7],'lightfm_hybrid')),
                    logging.StreamHandler(sys.stdout)
                ])

    def csv_to_df(self) -> tuple:
        df = pd.read_csv(self.rpath, sep='\t')
        df_user_features, df_item_features, df_interactions = df[['user_id', 'item_cashtags']], df[['item_id', 'item_timestamp', 'item_body','item_titles', 'item_cashtags', 'item_industries', 'item_sectors']], df[['user_id', 'item_id']]
        return (df_user_features, df_item_features, df_interactions)

    def build_id_mappings(self, df_interactions: pd.DataFrame, df_item_features: pd.DataFrame) -> Dataset:
        item_industries = list(set('|'.join(df_item_features['item_industries'].tolist()).split('|')))
        item_sectors = list(set('|'.join(df_item_features['item_sectors'].tolist()).split('|')))
        item_cashtags = list(set('|'.join(df_item_features['item_cashtags'].tolist()).split('|')))
        
        dataset = Dataset()
        dataset.fit((x for x in df_interactions['user_id']), 
                    (x for x in df_interactions['item_id']),
                    item_features=item_industries+item_sectors+item_cashtags)
        return (dataset, item_industries, item_sectors, item_cashtags)

    def build_interactions_matrix(self, dataset: Dataset, df_interactions: pd.DataFrame) -> tuple:
        def gen_rows(df):
            for row in df.itertuples(index=False):
                yield row
                
        (interactions, weights) = dataset.build_interactions(gen_rows(df_interactions))
        return (interactions, weights)

    def build_item_features(self, dataset: Dataset, df_item_features: pd.DataFrame) -> csr_matrix:
        def gen_rows(df):
            for row in df.itertuples(index=False):
                d = row._asdict()
                item_industries = d['item_industries'].split('|') if '|' in d['item_industries'] else [d['item_industries']]
                item_sectors = d['item_sectors'].split('|') if '|' in d['item_sectors'] else [d['item_sectors']]
                item_cashtags = d['item_cashtags'].split('|') if '|' in d['item_cashtags'] else [d['item_cashtags']]
                yield [d['item_id'], item_industries+item_sectors+item_cashtags]
        
        item_features = dataset.build_item_features(gen_rows(df_item_features))
        return item_features

    def cross_validate_interactions(self, interactions: coo_matrix) -> tuple:
        train, test = random_train_test_split(interactions)
        return train, test

    def cf_model_pure(self, train: coo_matrix, params: tuple) -> LightFM:
        logger = logging.getLogger()
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

        # Run 3 epochs and time it.
        logger.info('Begin fitting collaborative filtering model...')
        model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
        return model

    def hybrid_model(self, params: tuple, train: coo_matrix, item_features: csr_matrix) -> LightFM:
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
                        epochs=NUM_EPOCHS,
                        num_threads=NUM_THREADS)
        return model

    def evaluate_model(self, model: LightFM, model_name: str, eval_metrics: list, sets: tuple, NUM_THREADS: str, item_features: csr_matrix=None, k: int=None):
        logger = logging.getLogger()
        train, test = sets
        model_name = 'Collaborative Filtering' if model_name == 'cf' else 'Hybrid'

        def auc():
            train_auc = auc_score(model,
                            train,
                            item_features=item_features if item_features is not None else None,
                            num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set AUC: %s' % train_auc)

            test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    item_features=item_features if item_features is not None else None,
                    num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set AUC: %s' % test_auc)

        def precrec():
            train_precision = precision_at_k(model, 
                                train, 
                                k=k, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set Precision@%s: %s' % (k, train_precision))
            test_precision = precision_at_k(model, 
                                test, 
                                k=k, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set Precision@%s: %s' % (k, test_precision))

            train_recall = recall_at_k(model, 
                                train, 
                                k=k, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set Recall@%s: %s' % (k, train_recall))
            test_recall = recall_at_k(model, 
                                test, 
                                k=k, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set Recall@%s: %s' % (k, test_recall))

            f1_train, f1_test = 2*(train_recall * train_precision) / (train_recall + train_precision), 2*(test_recall * test_precision) / (test_recall + test_precision)
            logger.info(model_name+' training set F1 Score: %s' % (f1_train))
            logger.info(model_name+' test set F1 Score: %s' % (f1_test))

        def mrr():
            train_mrr = reciprocal_rank(model, 
                                train, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' training set MRR: %s' % (train_mrr))
            test_mrr = reciprocal_rank(model, 
                                test, 
                                item_features=item_features if item_features is not None else None, 
                                num_threads=NUM_THREADS).mean()
            logger.info(model_name+' test set MRR: %s' % (test_mrr))

        for metric in eval_metrics:
            locals()[metric]()

    def run(self):
        self.logger()
        logger = logging.getLogger()
        params = (NUM_THREADS, _, _, _) = (4,30,3,1e-16)

        df_user_features, df_item_features, df_interactions = self.csv_to_df()
        dataset, item_industries, item_sectors, item_cashtags = self.build_id_mappings(df_interactions, df_item_features)
        
        interactions, _ = self.build_interactions_matrix(dataset, df_interactions)
        item_features = self.build_item_features(dataset, df_item_features)
        train, test = self.cross_validate_interactions(interactions)

        logger.info('The dataset has %s users and %s items with %s interactions in the test and %s interactions in the training set.' % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        cf_model = self.cf_model_pure(train, params)
        self.evaluate_model(model=cf_model, model_name='cf', eval_metrics=['auc'], sets=(train, test), NUM_THREADS=NUM_THREADS)

        logger.info('There are {0} distinct industries, {1} distinct sectors and {2} distinct cashtags.'.format(len(item_industries), len(item_sectors), len(item_cashtags)))

        hybrid_model = self.hybrid_model(params, train, item_features)
        self.evaluate_model(model=hybrid_model, model_name='h', eval_metrics=['auc', 'precrec', 'mrr'], sets=(train, test), NUM_THREADS=NUM_THREADS, item_features=item_features, k=10)



if __name__=="__main__":
    bm = UserCashtagBaselineModel()
    bm.run()