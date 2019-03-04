from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re, spacy, json, logging, multiprocessing, sys
from datetime import datetime

from utils.misc import Utils

class TaggedDocumentIterator(object):
    def __init__(self, trainables_file):
        self.tf = trainables_file

    def __iter__(self):
        with open(self.tf) as f:
            for line in f:
                words, tags = line.split("\t")
                yield TaggedDocument(words=words.split(","), tags=tags.split(","))


class EmbeddingTrainer:
    def __init__(self):
        self.rpath = './data/csv/metadata_clean.csv'
        self.mpath = './models/doc2vec/'
        self.logpath = './log/d2v/'
        self.data = self.csv_to_dataframe()

    def logger(self):
        """ Sets the logger configuration to report to both std.out and to log to ./log/io/csv/
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1} ({2}).log".format(self.logpath, str(datetime.now())[:-7],'doc2vec')),
                    logging.StreamHandler(sys.stdout)
                ])

    def csv_to_dataframe(self) -> pd.DataFrame:
        """Reads in CSV file declared in __init__ (self.rpath) and converts it to a number of Pandas DataFrames.
            
        Returns:
            pandas.DataFrame: Returns metadata CSV file as pandas DataFrame.

        """

        logger = logging.getLogger()
        data = pd.read_csv(self.rpath, delimiter='\t')
        logger.info("Read file with {0} entries".format(len(data.index)))
        return data[['item_id', 'item_body', 'item_sectors', 'item_industries']]

    def write_trainable_input_file(self):
        logger, utils = logging.getLogger(), Utils()
        write_path = './data/trainable/'+str(datetime.now())[:-7]+'.txt'
        logging.info("Began writing trainables file ({0})".format(write_path))
        with open(write_path, 'w') as f:
            for _, row in self.data.iterrows():
                item_id = row['item_id']
                tokens = utils.tokenise(row['item_body'])
                item_sectors = list(map('SECTOR:{0}'.format, row['item_sectors'].split('|') if '|' in row['item_sectors'] else [row['item_sectors']]))
                item_industries =  list(map('INDUSTRY:{0}'.format, row['item_industries'].split('|') if '|' in row['item_industries'] else [row['item_industries']]))

                words = ",".join([str(x) for x in tokens])
                tags = ",".join([str(x) for x in [item_id]+item_sectors+item_industries])
                line = words+"\t"+tags
                f.write(line+"\n")
        logging.info("Finished writing trainables file ({0})".format(write_path))
                

    def train_model(self, tagged_docs):
        logger = logging.getLogger()
        no_of_workers = multiprocessing.cpu_count()/2

        model = Doc2Vec(vector_size=100,
                min_count=2,
                dm=1,
                workers=no_of_workers,
                epochs=20)
        
        logger.info("Began building vocabulary...")
        model.build_vocab(tagged_docs, progress_per=100000)
        logger.info("Finished building vocabulary.")

        logger.info("Began training model...")
        model.train(tagged_docs,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
        logger.info("Finished training model.")

        model.save(self.mpath+str(datetime.now())[:-7]+'.model')
    
    def run(self):
        self.logger()
        tagged_docs = TaggedDocumentIterator('./data/trainable/2019-03-04 20:14:37.txt')
        self.train_model(tagged_docs)


if __name__ == "__main__":
    et = EmbeddingTrainer()
    et.run()