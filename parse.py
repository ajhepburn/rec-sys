import os, sys, json, spacy, pymongo, logging
from datetime import timedelta, datetime

class Parser:
    def __init__(self, fp):
        if not os.path.isdir(fp):
            raise OSError('ERROR: Directory does not exist.')
        self.path = fp
        self.files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        self.files.sort()
        with open('./utilities/slang.txt') as sl:
            slang_terms = json.loads(sl.readline())
            self.slang_terms = [t.lower() for t in slang_terms]
        self.nlp = spacy.load('en')
        self.log_path = './log/'

    def test_db_conn(self):
        pass

    def tokenize(self):
        pass

    def extract_features(self, inputFile):
        pass

    def build_wlist(self):
        pass

    def db_insert(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["rec-sys"]
        col = db["messages"]

        logging.basicConfig(filename=self.log_path+'database/insertion_'+str(datetime.now())+'.log',level=logging.INFO)

        for filename in self.files:
            logging.info("File: "+filename)
            with open(self.path+filename) as inputFile:  
                for line in inputFile:
                    data = json.loads(line)
                    col.insert_one(data)

        # CLEAR LOG CONFIGS
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

if __name__ == "__main__":
    pd = Parser("/media/ntfs/st_2017/")
    pd.db_insert()
