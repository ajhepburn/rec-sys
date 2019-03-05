from datetime import datetime
import logging, sys
import pandas as pd
from spotlight.interactions import Interactions

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

    def build_interactions(self, df_interactions: pd.DataFrame) -> Interactions: return Interactions(df_interactions['user_id'], df_interactions['item_cashtags'])

    def run(self):
        self.logger()
        df = self.csv_to_df()
        df_interactions = df[['user_id', 'item_cashtags']]
        df_interactions = df_interactions.groupby(['user_id','item_cashtags']).size().reset_index() \
                                               .rename(columns={0:'interactions'})
        interactions = self.build_interactions(df_interactions)

if __name__ == "__main__":
    sim = SpotlightImplicitModel()
    sim.run()