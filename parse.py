import os, json, sys, logging, csv
from datetime import datetime


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
        industries, sectors, cashtags = [], [], []

        if symbols:                     # Checks to see if cashtags exist in tweet
            for s in symbols:
                industry, sector = s.get('industry'), s.get('sector')  # Check to see if a cashtag contains an 'Industry' tag, otherwise skip
                if not industry:
                    continue

                uid, uloc, iid, ii, isec, ic = d['user']['id'], d['user']['location'],d['id'], industry, sector, s['symbol']
                industries.append(ii)
                sectors.append(isec)
                cashtags.append(ic)

        return (uid, uloc, iid, industries, sectors, cashtags) if industries else ((None),)*6
                 
    def file_writer(self):
        """ Responsible for writing to ct_industry.csv in ./data/csv/ and logging each file read.
        Passes single line into self.parse which returns a tuple of metadata, this is then written
        to a single row in the CSV, provided the tuple returned by self.parse does not contain any None values.

        """

        logger = logging.getLogger()

        with open (os.path.join(self.wpath, 'metadata.csv'), 'w', newline='') as stocktwits_csv:
            fields = ['user_id', 'location','item_id', 'item_industries','item_sectors','item_cashtags']
            writer = csv.DictWriter(stocktwits_csv, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            line_count = 0

            for fp in self.files:
                logger.info('Reading file {0}'.format(fp))
                with open(os.path.join(self.rpath, fp)) as f:
                    for l in f:
                        if not all(self.parse(l)):
                            continue
                        uid, uloc, iid, ii, isec, ic = self.parse(l)
                        writer.writerow({'user_id':uid, 'location':uloc,'item_id':iid, 'item_industries':ii,'item_sectors':isec,'item_cashtags':ic})
                        line_count+=1
        
        return line_count



    def run(self):
        self.logger()
        logger = logging.getLogger()
        logger.info("Starting CSV write")

        line_count = self.file_writer()

        logger.info("Finished CSV write with {0} documents".format(line_count))




if __name__ == "__main__":
    ab = AttributeParser('2017_02_01')
    ab.run()