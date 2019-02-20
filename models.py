import json, csv, sys, time, os, torch, logging, logging.handlers
from os.path import join, isfile, exists
from itertools import islice
from datetime import datetime
import numpy as np

from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from spotlight.interactions import Interactions
# from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import mrr_score


class LFMBaselines:
    def __init__(self):
        self.csvpath = './data/csv/'
        self.logpath = './log/models/'
    
    def logger(self, loss):
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} LFM ({2}).log".format(self.logpath, str(datetime.now())[:-7],loss)),
                    logging.StreamHandler(sys.stdout)
                ])

    def get_data(self):
        stocktwits = join(self.csvpath, 'stocktwits.csv')
        if not exists(stocktwits): raise Exception('Missing Stocktwits File')
        return csv.DictReader(open(stocktwits), delimiter='\t')

    def build_dataset(self):
        logger = logging.getLogger()
        dataset = Dataset()
        logger.info("Building Dataset")
        dataset.fit((x['user_id'] for x in self.get_data()),
                    (x['item_id'] for x in self.get_data()),
                    item_features=(x['item_cashtags'] for x in self.get_data()))

        num_users, num_items = dataset.interactions_shape()
        logger.info('Dataset Build Complete, No. Users: {}, No. Items: {}.'.format(num_users, num_items))
 
        (interactions, weights) = dataset.build_interactions(((x['user_id'], x['item_id'])
                                                      for x in self.get_data()))
        logger.info('Built Interactions Matrix')

        item_features = dataset.build_item_features(((x['item_id'], [x['item_cashtags']])
                                              for x in self.get_data()))
        logger.info('Build Item Features')
        return interactions, item_features

    def evaluate_model(self, model, k, item_features, train, test):
        logger = logging.getLogger()

        #MAP@K
        train_precision = precision_at_k(model, train, k=k, item_features=item_features, num_threads=4).mean()
        logger.info('Train Precision@{} Complete'.format(k))
        test_precision = precision_at_k(model, test, k=k, item_features=item_features, num_threads=4).mean()
        logger.info('Test Precision@{} Complete'.format(k))

        #MAR@K
        train_recall = recall_at_k(model, train, k=k, item_features=item_features, num_threads=4).mean()
        logger.info('Train Recall@{} Complete'.format(k))
        test_recall = recall_at_k(model, test, k=k, item_features=item_features, num_threads=4).mean()
        logger.info('Test Recall@{} Complete'.format(k))

        #AUROC
        train_auc = auc_score(model, train, item_features=item_features, num_threads=4).mean()
        logger.info('Train AUROC Complete')
        test_auc = auc_score(model, test, item_features=item_features, num_threads=4).mean()
        logger.info('Test AUROC Complete')

        logger.info('Precision: train %.6f, test %.6f.' % (train_precision, test_precision))
        logger.info('Recall: train %.6f, test %.6f.' % (train_recall, test_recall))
        logger.info('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    def run(self, lf):
        self.logger(lf)
        logger = logging.getLogger()
        start_time = time.time()
        interactions, item_features = self.build_dataset()
        train, test = random_train_test_split(interactions)
        logger.info('Fitting Model... Loss: {}'.format(lf.upper()))
        model = LightFM(learning_rate=0.05, loss=lf)
        model.fit(train, item_features=item_features, epochs=10, num_threads=4)
        logger.info('Model fitting completed')
        self.evaluate_model(model, 3, item_features, train, test)
        logger.info("--- %s seconds ---" % (time.time() - start_time))


class SLBaselines:
    def __init__(self):
        self.csvpath = './data/csv/'
        self.logpath = './log/models/'

    def logger(self, loss):
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} SL ({2}).log".format(self.logpath, str(datetime.now())[:-7],loss)),
                    logging.StreamHandler(sys.stdout)
                ])

    def get_interactions(self):
        data = np.genfromtxt(os.path.join(self.csvpath, 'stocktwits.csv'), delimiter='\t', names=True, usecols=(0,1),dtype=(int, int))
        return Interactions(data['user_id'], data['item_id'])

    def run(self, lf):
        self.logger(lf)
        logger = logging.getLogger()
        # start_time = time.time()

        logger.info('Building Interactions Matrix...')
        dataset = self.get_interactions()
        logger.info(dataset)
        
        train, test = random_train_test_split(dataset)

        logger.info('Fitting Model... Loss: {}'.format(lf.upper()))
        model = ImplicitFactorizationModel(loss=lf,
                                            n_iter=10,
                                            use_cuda=True)
        model.fit(train)
        logging.info('Model Fitting Completed')

        # mrr = mrr_score(model, test)

if __name__ == "__main__":
    cfm = LFMBaselines()
    cfm.run('bpr')
    # slb = SLBaselines()
    # slb.run('bpr')