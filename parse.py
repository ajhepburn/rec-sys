import os, json, sys, logging, csv
from datetime import datetime
import pandas as pd


class AttributeParser:
    def __init__(self, limit: str):
        self.rpath = '/media/ntfs/st_2017'
        self.wpath = './data/csv/'
        self.logpath = './log/io/'

        self.files = [f for f in os.listdir(self.rpath) if os.path.isfile(os.path.join(self.rpath, f))]
        self.files.sort()
        self.files = self.files[:next(self.files.index(x) for x in self.files if x.endswith(limit))]

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1}_log ({2}).log".format(self.logpath, str(datetime.now())[:-7],'ct_industry_csv')),
                    logging.StreamHandler(sys.stdout)
                ])
    
    def parse(self, l) -> tuple:
        """ Takes in a single line and attempts to retrieve User ID, Item ID, list of Industry Tags, list of Cashtags.
        Will return a tuple filled with None values if:
            * No cashtags are found with at least one industry tag.
        If there are no cashtags at all in a single tweet, the line will be skipped.

        """

        d = json.loads(l)['data']
        symbols = d.get('symbols', False)
        titles, cashtags, industries, sectors = [], [], [], []

        if symbols:                     # Checks to see if cashtags exist in tweet
            for s in symbols:
                s_title, s_symbol, s_industry, s_sector = s.get('title'), s.get('symbol'), s.get('industry'), s.get('sector')  # Check to see if a cashtag contains an 'Industry' tag, otherwise skip
                if not s_industry:
                    continue

                user_id, user_location = d.get('user').get('id'), d.get('user').get('location')
                item_id, item_timestamp = d.get('id'), d.get('created_at')

                titles[len(titles):], cashtags[len(cashtags):], industries[len(industries):], sectors[len(sectors):] = tuple(zip((s_title, s_symbol, s_industry, s_sector)))

        return (user_id, user_location, item_id, item_timestamp, titles, cashtags, industries, sectors) if industries else ((None),)*8
                 
    def file_writer(self):
        """ Responsible for writing to ct_industry.csv in ./data/csv/ and logging each file read.
        Passes single line into self.parse which returns a tuple of metadata, this is then written
        to a single row in the CSV, provided the tuple returned by self.parse does not contain any None values.

        """

        logger = logging.getLogger()

        with open (os.path.join(self.wpath, 'metadata.csv'), 'w', newline='') as stocktwits_csv:
            fields = ['user_id', 'item_id', 'user_location', 'item_timestamp', 'item_titles','item_cashtags','item_industries','item_sectors']
            writer = csv.DictWriter(stocktwits_csv, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            line_count = 0

            for fp in self.files:
                logger.info('Reading file {0}'.format(fp))
                with open(os.path.join(self.rpath, fp)) as f:
                    for l in f:
                        if not all(self.parse(l)):
                            continue
                        user_id, user_location, item_id, item_timestamp, titles, cashtags, industries, sectors = self.parse(l)
                        writer.writerow({'user_id':user_id, 'item_id':item_id, 'user_location':user_location, 'item_timestamp':item_timestamp, 'item_titles':','.join(map(str, titles)),'item_cashtags':','.join(map(str, cashtags)), 'item_industries':','.join(map(str, industries)),'item_sectors':','.join(map(str, sectors))})
                        line_count+=1
        
        return line_count



    def run(self):
        self.logger()
        logger = logging.getLogger()
        logger.info("Starting CSV write")

        line_count = self.file_writer()

        logger.info("Finished CSV write with {0} documents".format(line_count))



class AttributeCleaner:
    def __init__(self):
        self.rpath = './data/csv/metadata.csv'
        self.df = self.csv_to_dataframe()

    def csv_to_dataframe(self):
        """ Reads in CSV and converts to Pandas DataFrame, outputting number of entries to stdout.

        """

        data = pd.read_csv(self.rpath, delimiter='\t')
        print("Read file with {0} entries".format(len(data.index)))
        return data

    def dataframe_to_csv(self):
        """ Writes cleaned dataset back to CSV format.

        """

        self.df.to_csv(path_or_buf='./data/csv/metadata_clean.csv', index=False, sep='\t')
        print("Written CSV at {0} with {1} entries".format(str(datetime.now())[:-7], len(self.df.index)))

    # def clean_user_locations(self):
    #     print(self.df)

    def clean_rare_users(self):
        """ Removes users who have made less than k tweets, outputting change of entry count
            to stdout.
        """

        data_count, k = len(self.df.index), 20
        cleaned_users = self.df.groupby('user_id').filter(lambda x: len(x) > k)
        print("Removed users with less than {0} tweets. Size of DataFrame: {1} -> {2}".format(k, data_count, len(cleaned_users.index)))
        self.df = cleaned_users


    def run(self):
        self.clean_rare_users()
        self.dataframe_to_csv()
        # self.clean_user_locations()






if __name__ == "__main__":
    ab = AttributeParser('2017_02_01')
    ab.run()
    ac = AttributeCleaner()
    ac.run()