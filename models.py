import json, csv
from os.path import join, isfile
from itertools import islice

class CFModels:
    def __init__(self):
        self.csvpath = './data/csv/'

    def extract_features(self, feature: str):
        if feature not in ('user', 'item'):
            raise Exception("Unrecognised Features Type: "+feature)
        
        ffile = join(self.csvpath, feature+'_features.csv')
        if not isfile(ffile):
            raise Exception("Missing Features File")

        def extract_user_features(dr: csv.DictReader):
            for line in islice(dr, 1):
                print(json.dumps(line, indent=4))

        def extract_item_features(dr: csv.DictReader):
            for line in islice(dr, 1):
                print(json.dumps(line, indent=4))
        
        dr = csv.DictReader(open(ffile),delimiter='\t')
        locals()["extract_"+feature+"_features"](dr)

if __name__ == "__main__":
    cfm = CFModels()
    cfm.extract_features('user')