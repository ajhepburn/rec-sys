from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re, spacy, json, logging

class TaggedDocumentIterator(object):
    def __init__(self, **kwargs):
        if kwargs is None:
            raise('Nothing passed to iterator.')
        self.tweet = kwargs

    def __iter__(self):
        pass
            # for line in f:
            #     s_line = line.strip('\n')
            #     tokens = s_line.split(" ")
            #     tag = tokens[0]
            #     words = tokens[1:]
            #yield TaggedDocument(words=words, tags=[tag])

class EmbeddingParser:
    def __init__(self):
        self.rpath = './data/csv/metadata_clean.csv'

    def csv_to_dataframe(self) -> pd.DataFrame:
        """Reads in CSV file declared in __init__ (self.rpath) and converts it to a number of Pandas DataFrames.
            
        Returns:
            pandas.DataFrame: Returns metadata CSV file as pandas DataFrame.

        """

        logger = logging.getLogger()
        data = pd.read_csv(self.rpath, delimiter='\t')
        logger.info("Read file with {0} entries".format(len(data.index)))
        return data
    
    def run(self):
        data = self.csv_to_dataframe()
        # for idx, row in data.iterrows():
        #     tokens = self.tokenise(row['item_body'])
        #     print(tokens)
            #tagged_docs = TaggedDocumentIterator()


if __name__ == "__main__":
    ep = EmbeddingParser()
    ep.run()