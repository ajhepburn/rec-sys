import json, csv, sys
from os.path import join, isfile, exists
from itertools import islice
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from datetime import datetime

class CFModels:
    def __init__(self):
        self.csvpath = './data/csv/'

    def get_data(self):
        stocktwits = join(self.csvpath, 'stocktwits.csv')
        if not exists(stocktwits): raise Exception('Missing Stocktwits File')
        return csv.DictReader(open(stocktwits), delimiter='\t')

    def build_dataset(self):
        dataset = Dataset()
        print("Building Dataset ({})".format(str(datetime.now())[:-7]))
        dataset.fit((x['user_id'] for x in self.get_data()),
                    (x['item_id'] for x in self.get_data()),
                    item_features=(x['item_cashtags'] for x in self.get_data()))

        num_users, num_items = dataset.interactions_shape()
        print('Dataset Build Complete ({})\nNo. Users: {}, No. Items: {}.'.format(str(datetime.now())[:-7], num_users, num_items))
 
        (interactions, weights) = dataset.build_interactions(((x['user_id'], x['item_id'])
                                                      for x in self.get_data()))
        print('Built Interactions Matrix ({})'.format(str(datetime.now())[:-7]))

        item_features = dataset.build_item_features(((x['item_id'], [x['item_cashtags']])
                                              for x in self.get_data()))
        print('Build Item Features ({})'.format(str(datetime.now())[:-7]))
        return interactions, item_features

    def evaluate_model(self, model, item_features, train, test):
        train_precision = precision_at_k(model, train, k=10, item_features=item_features, num_threads=4).mean()
        test_precision = precision_at_k(model, test, k=10, item_features=item_features, num_threads=4).mean()
 
        train_auc = auc_score(model, train, item_features=item_features, num_threads=4).mean()
        test_auc = auc_score(model, test, item_features=item_features, num_threads=4).mean()

        print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
        print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    def run(self, lf):
        interactions, item_features = self.build_dataset()
        train, test = random_train_test_split(interactions)
        print('Fitting Model... ({}) Loss: {}'.format(str(datetime.now())[:-7],lf.upper()))
        model = LightFM(learning_rate=0.05, loss=lf)
        model.fit(train, item_features=item_features, epochs=10)
        print('Model fitting completed ({})'.format(str(datetime.now())[:-7]))
        self.evaluate_model(model, item_features, train, test)
        

if __name__ == "__main__":
    cfm = CFModels()
    cfm.run('bpr')