import os, csv, sys, random
from collections import defaultdict
from scipy import sparse

class MetadataTensorRec:
    def __init__(self):
        self.rpath = './data/csv/metadata.csv'

    def load_csv(self, rpath):
        with open(self.rpath, 'r') as metadata_file:
            metadata_file_reader = csv.reader(metadata_file)
            raw_metadata = list(metadata_file_reader)
            raw_metadata_header = raw_metadata.pop(0)
        return (raw_metadata, raw_metadata_header)

    def map_to_internal_ids(self, raw_metadata):
        pass

    def run(self, rpath):
        raw_metadata, raw_metadata_header = self.load_csv(rpath)

if __name__ == "__main__":
    mtr = MetadataTensorRec()
    mtr.run('./data/csv/metadata.csv')
        


