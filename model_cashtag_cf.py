import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM

import sys, regex

class UserCashtagBaselineModel:
    def __init__(self):
        self.logpath = './log/models/user_ct_base'
        self.rpath = './data/csv/metadata_clean.csv'

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
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

        # Run 3 epochs and time it.
        model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
        return model

    def hybrid_model(self, params: tuple, train: coo_matrix, item_features: csr_matrix) -> LightFM:
        NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA = params
        # Define a new model instance
        model = LightFM(loss='warp',
                        item_alpha=ITEM_ALPHA,
                        no_components=NUM_COMPONENTS)

        # Fit the hybrid model. Note that this time, we pass
        # in the item features matrix.
        model = model.fit(train,
                        item_features=item_features,
                        epochs=NUM_EPOCHS,
                        num_threads=NUM_THREADS)
        return model

    def run(self):
        params = (NUM_THREADS, NUM_COMPONENTS, NUM_EPOCHS, ITEM_ALPHA) = (4,30,3,1e-16)

        df_user_features, df_item_features, df_interactions = self.csv_to_df()
        dataset, item_industries, item_sectors, item_cashtags = self.build_id_mappings(df_interactions, df_item_features)
        
        interactions, _ = self.build_interactions_matrix(dataset, df_interactions)
        item_features = self.build_item_features(dataset, df_item_features)
        train, test = self.cross_validate_interactions(interactions)

        print('The dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

        cf_model = self.cf_model_pure(train, params)
        train_auc = auc_score(cf_model, train, num_threads=NUM_THREADS).mean()
        print('Collaborative filtering train AUC: %s' % train_auc)

        test_auc = auc_score(cf_model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
        print('Collaborative filtering test AUC: %s' % test_auc)

        print('There are {0} distinct industries, {1} distinct sectors and {2} distinct cashtags.'.format(len(item_industries), len(item_sectors), len(item_cashtags)))

        hybrid_model = self.hybrid_model(params, train, item_features)
                # Don't forget the pass in the item features again!
        train_auc = auc_score(hybrid_model,
                            train,
                            item_features=item_features,
                            num_threads=NUM_THREADS).mean()
        print('Hybrid training set AUC: %s' % train_auc)

        test_auc = auc_score(hybrid_model,
                    test,
                    train_interactions=train,
                    item_features=item_features,
                    num_threads=NUM_THREADS).mean()
        print('Hybrid test set AUC: %s' % test_auc)

        k = 10
        train_precision = precision_at_k(hybrid_model, 
                            train, 
                            k=k, 
                            item_features=item_features, 
                            num_threads=4).mean()
        print('Hybrid training set Precision@%s: %s' % (k, train_precision))
        test_precision = precision_at_k(hybrid_model, 
                            test, 
                            k=k, 
                            item_features=item_features, 
                            num_threads=4).mean()
        print('Hybrid test set Precision@%s: %s' % (k, test_precision))

        #MAR@K
        train_recall = recall_at_k(hybrid_model, 
                            train, 
                            k=k, 
                            item_features=item_features, 
                            num_threads=4).mean()
        print('Hybrid training set Recall@%s: %s' % (k, train_recall))
        test_recall = recall_at_k(hybrid_model, 
                            test, 
                            k=k, 
                            item_features=item_features, 
                            num_threads=4).mean()
        print('Hybrid training set Recall@%s: %s' % (k, test_recall))



if __name__=="__main__":
    bm = UserCashtagBaselineModel()
    bm.run()