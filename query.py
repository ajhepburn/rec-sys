from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON

import pandas as pd
import more_itertools as mit

import logging
import sys
import time

class Queryer:
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

    def run_query(self, symbols: str):
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
                ?item
                ?title
                ?itemDescription
                ?exchangeLabel
                ?symbol
                WHERE 
                {{
                ?item p:P414 ?exchanges.
                ?exchanges ps:P414 ?exchange.
                {{ ?exchanges pq:P249 ?symbol. }} 
                    UNION {{ ?item wdt:P249 ?symbol }} .
                ?item rdfs:label ?title . 
                FILTER(REGEX(?symbol, "{}" ))
                FILTER(lang(?title) = "en")
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
            """.format(symbols)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.DataFrame()

        try:
            if results['results']['bindings']:
                results_df = pd.io.json.json_normalize(results['results']['bindings'])
                results_df['exchangeLabel.value'] = results_df['exchangeLabel.value'].str.replace("New York Stock Exchange","NYSE")
                results_df = results_df[['symbol.value','exchangeLabel.value','itemDescription.value']]
                results_df = results_df.rename(columns={
                    'symbol.value':'symbol', 
                    'exchangeLabel.value':'exchange', 
                    'itemDescription.value': 'wdDescription',
                    'item.value':'wdEntity'
                })
        except KeyError: return None
        return results_df

    def run(self):
        df = pd.read_csv(self._rpath+'tag_cat_cashtags_clean.csv', sep='\t')
        df = df[~df.exchange.str.contains('NYSEArca')]
        df = df[['id', 'title', 'target', 'symbol', 'exchange']]
        symbols = ['^'+s+'$' for s in df.symbol.tolist()]
        symbols_split = [list(c) for c in mit.divide(8, symbols)]
        results = pd.DataFrame()
        for i, s in enumerate(symbols_split):
            print("Parsing symbol list of size: {}, List: {}/{}".format(len(s), i+1, len(symbols_split)))
            symbols = '|'.join(s)
            query_results = self.run_query(symbols)
            results = results.append(query_results)
        print("Size of results returned: {}".format(results.shape[0]))


        #symbols = '|'.join(symbols)
        print("Number of cashtags {}".format(df.shape[0]))
        # query_results = self.run_query(symbols)
        # print(query_results)

        # print("Number of entries before description processing: {}".format(df.shape[0]))
        # df['description'] = ''
        # c = 0
        # for i, row in df.iterrows():
        #     desc = self.run_query(row.symbol, row.exchange)
        #     print(row, "\n", desc)
        #     sys.exit(0)
        #     if not desc:
        #         print("Row: {}, Hit Count: {}".format(i, c))
        #         continue
        #     c += 1
        #     print("Row: {}, Hit Count: {}".format(i, c))
        #     df.at[i, 'description'] = desc
        
        # print(df.head())
        # sys.exit(0)
        # print("Number of entries after description processing: {}".format(df.shape[0]))

if __name__ == "__main__":
    queryer = Queryer()
    queryer.run()