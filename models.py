import json, csv, sys
from os.path import join, isfile, exists
from itertools import islice
from lightfm.data import Dataset
from lightfm import LightFM

class CFModels:
    def __init__(self):
        self.csvpath = './data/csv/'

    def get_data(self):
        stocktwits = join(self.csvpath, 'stocktwits.csv')
        if not exists(stocktwits): raise Exception('Missing Stocktwits File')
        return csv.DictReader(open(stocktwits), delimiter='\t')

    def build_dataset(self):
        dataset = Dataset()
        dataset.fit((x['user_id'] for x in self.get_data()),
                    (x['item_id'] for x in self.get_data()),
                    item_features=(x['item_cashtags'] for x in self.get_data()))

        num_users, num_items = dataset.interactions_shape()
        print('Model Fitting Complete\n','Num users: {}, num_items {}.'.format(num_users, num_items))
 
        (interactions, weights) = dataset.build_interactions(((x['user_id'], x['item_id'])
                                                      for x in self.get_data()))

        item_features = dataset.build_item_features(((x['item_id'], [x['item_cashtags']])
                                              for x in self.get_data()))
        return interactions, item_features

    def run(self, lf):
        interactions, item_features = self.build_dataset()
        model = LightFM(loss=lf)
        model.fit(interactions, item_features=item_features)

if __name__ == "__main__":
    cfm = CFModels()
    cfm.run('bpr')