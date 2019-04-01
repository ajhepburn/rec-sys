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
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        # query = """
        #     PREFIX dbp: <http://dbpedia.org/property/>
        #     PREFIX dbo: <http://dbpedia.org/ontology/>
        #     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        #     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        #     SELECT *
        #     WHERE {{
        #     ?s1 dbp:symbol ?symbol .
        #     ?s1 rdf:type ?exchange .
        #     ?s1 rdfs:comment ?comment .
        #     ?symbol bif:contains  '"{}"'  .
        #     FILTER(lang(?comment) = "en").
        #     FILTER(regex(str(?exchange), "{}" ) )
        #     }}
        # """.format(symbol, exch)
        query = """
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wds: <http://www.wikidata.org/entity/statement/>
                PREFIX wdv: <http://www.wikidata.org/value/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX p: <http://www.wikidata.org/prop/>
                PREFIX ps: <http://www.wikidata.org/prop/statement/>
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX bd: <http://www.bigdata.com/rdf#>

                SELECT
                ?title
                ?itemDescription
                ?exchangeLabel
                ?symbol
                WHERE 
                {
                ?item p:P414 ?exchanges.
                ?exchanges ps:P414 ?exchange.
                { ?exchanges pq:P249 ?symbol. } 
                    UNION { ?item wdt:P249 ?symbol } .
                ?item rdfs:label ?title . 
                FILTER(REGEX(?symbol, "AAPL|GOOG|TWTR" ))
                FILTER(lang(?title) = "en")
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                }
            """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        # for result in results["results"]["bindings"]:
        print(results)
        sys.exit(0)
        result = None
        try:
            if results['results']['bindings']:
                result = results['results']['bindings'][0]['comment']['value']
        except KeyError: pass
        return result

    def run(self):
        df = pd.read_csv(self._rpath+'tag_cat_cashtags_clean.csv', sep='\t')
        df = df[~df.exchange.str.contains('NYSE')]
        print("Number of entries before description processing: {}".format(df.shape[0]))
        df['description'] = ''
        c = 0
        for i, row in df.iterrows():
            desc = self.run_query(row.symbol, row.exchange)
            print(row, "\n", desc)
            sys.exit(0)
            if not desc:
                print("Row: {}, Hit Count: {}".format(i, c))
                continue
            c += 1
            print("Row: {}, Hit Count: {}".format(i, c))
            df.at[i, 'description'] = desc
        
        print(df.head())
        sys.exit(0)
        print("Number of entries after description processing: {}".format(df.shape[0]))

if __name__ == "__main__":
    queryer = DBPediaQueryer()
    queryer.run()