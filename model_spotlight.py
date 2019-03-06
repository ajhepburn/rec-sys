from datetime import datetime
import logging, sys
import pandas as pd
from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import precision_recall_score, mrr_score
import numpy as np

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
        return df

    def run(self):
        self.logger()
        df = self.csv_to_df()
        df_interactions = df[['user_id', 'item_tag_ids']]
        df_interactions = df_interactions.groupby(['user_id','item_tag_ids']).size().reset_index() \
                                               .rename(columns={0:'interactions'})
        user_ids, cashtag_ids = df_interactions['user_id'].values.astype(int), df_interactions['item_tag_ids'].values.astype(int)
        implicit_interactions = Interactions(user_ids, cashtag_ids)
        train, test = random_train_test_split(implicit_interactions)
        implicit_model = ImplicitFactorizationModel(use_cuda=True)
        implicit_model.fit(train, verbose=True)
        # predictions = implicit_model.predict(406225, item_ids=None)
        # # print(predictions)
        train_mrr = mrr_score(implicit_model, train).mean()
        test_mrr = mrr_score(implicit_model, test).mean()

        train_prec, train_rec = precision_recall_score(implicit_model, train)
        test_prec, test_rec = precision_recall_score(implicit_model, test)

        print('Train MRR {:.8f}, test MRR {:.8f}'.format(train_mrr, test_mrr))
        print('Train Precision@10 {:.8f}, test Precision@10 {:.8f}'.format(train_prec.mean(), test_prec.mean()))
        print('Train Recall@10 {:.8f}, test Recall@10 {:.8f}'.format(train_rec.mean(), test_rec.mean()))

if __name__ == "__main__":
    sim = SpotlightImplicitModel()
    sim.run()