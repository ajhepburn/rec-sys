from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON

import pandas as pd

import logging
import sys

class DBPediaQueryer:
    def __init__(self):
        self._logpath = './log/database/dbquery'
        self._rpath = './data/csv/'

    def logger(self):
        """Sets logger config to both std.out and log ./log/io/dbquery

        Also sets the formatting instructions for the log file, prints Time,
        Current Thread, Logging Type, Message.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("{0}/{1} ({2}).log".format(
                    self._logpath,
                    str(datetime.now())[:-7],
                    'dbquery',
                )),
                logging.StreamHandler(sys.stdout)
            ])

    def run_query(self, symbol: str, exch: str):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery("""
            PREFIX dbp: <http://dbpedia.org/property/>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT *
            WHERE {
            ?s1 dbp:symbol ?symbol .
            ?s1 dbp:tradedAs ?exchange .
            ?s1 rdfs:comment ?comment .
            ?symbol bif:contains  '"AAPL"'  .
            FILTER(lang(?comment) = "en").
            FILTER(regex(str(?exchange), "NASDAQ" ) )
            }
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            print(result['comment']['value'])

    def run(self):
        df = pd.read_csv(self._rpath+'tag_cat_cashtags_clean.csv', sep='\t')
        for _, row in df.iterrows():
            self.run_query(row.symbol, row.exchange)
            sys.exit(0)

if __name__ == "__main__":
    queryer = DBPediaQueryer()
    queryer.run()