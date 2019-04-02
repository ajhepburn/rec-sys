from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON

from utils.queries import queryFormatter
from Levenshtein import _levenshtein

import pandas as pd
import more_itertools as mit

import logging
import sys
import time
import os
import re

class Queryer:
    def __init__(self):
        self._logpath = './log/database/dbquery'
        self._rpath = './data/csv/'
        self._wpath = './data/txt/queryer/levenshtein/'

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

    def run_query(self, symbols: str, keyword: str):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = queryFormatter(keyword, symbols)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.DataFrame()

        try:
            if results['results']['bindings']:
                results_df = pd.io.json.json_normalize(results['results']['bindings'])
                results_df['exchangeLabel.value'] = results_df['exchangeLabel.value'].str.replace("New York Stock Exchange","NYSE")            
                results_df = results_df[['symbol.value','exchangeLabel.value','title.value','itemDescription.value', 'item.value']] if keyword is 'standard' else results_df[['exchangeLabel.value','title.value','itemDescription.value', 'item.value']]  
                results_df = results_df.rename(columns={
                    'symbol.value':'symbol', 
                    'exchangeLabel.value':'exchange', 
                    'title.value': 'title',
                    'itemDescription.value': 'wdDescription',
                    'item.value':'wdEntity'
                })
        except KeyError: return None
        return results_df

    def query_loop(self, df: pd.DataFrame, query: str):
        if query is 'standard':
            symbols = ['^'+s+'$' for s in df.symbol.tolist()]
        elif query is 'by_name':
            symbols = [s.strip() for s in df.title.tolist()]
            symbols = [re.sub(' +', ' ', s) for s in symbols]
        symbols_split = [list(c) for c in mit.divide(8, symbols)]

        results = pd.DataFrame()
        for i, s in enumerate(symbols_split):
            print("Parsing symbol list of size: {}, List: {}/{}".format(len(s), i+1, len(symbols_split)))
            symbols = '|'.join(s)
            query_results = self.run_query(symbols, query)
            results = results.append(query_results)
        print("Size of results returned: {}".format(results.shape[0]))
        return results

    def fuzzy_string_comparison(self, df: pd.DataFrame, df_main: pd.DataFrame):
        cashtags_main = df_main['title'].tolist()
        cashtags_main = [s.lower() for s in cashtags_main]
        for _, row in df.iterrows():
            title = row['title'].lower()
            ratios = {}
            for tag in cashtags_main:
                lev = _levenshtein.ratio(tag, title)
                ratios[tag] = lev
            matched_string = max(ratios, key=lambda key: ratios[key])
            print("Title in query DF: {}, Title in original DF: {}".format(title, matched_string))
            time.sleep(2)
        sys.exit(0)


    def clean_results(self, df: pd.DataFrame, keyword: str, df_main: pd.DataFrame=None):
        df = df.copy()
        df = df[df.exchange.str.contains('^NASDAQ$|^NYSE$')]
        regex_companies = re.compile(r'(?:\s+(?:Incorporated|Corporation|Company|Inc Common Stock|QQQ|ETF|PLC|SA|Inc|Corp|Ltd|LP|plc|Group|The|Co|Limited))+\s*$', flags=re.IGNORECASE)
        df['title'] = df['title'].str.replace('[^\w\s]','')
        df['title'] = df['title'].str.strip()
        df['title'] = df['title'].str.replace(' +|The ', ' ', regex=True)
        df['title'] = df['title'].str.replace(regex_companies, '')
        df['title'] = df['title'].str.strip()
        print("Size of results after cleaning exchanges: {}".format(df.shape[0]))
        if keyword is 'by_name':
            self.fuzzy_string_comparison(df, df_main)
        return df

    def run(self):
        df = pd.read_csv(self._rpath+'tag_cat_cashtags_clean.csv', sep='\t')
        # df = df[df.exchange.str.contains('^NASDAQ$|^NYSE$')]
        df = self.clean_results(df, 'standard')
        df = df[['id', 'title', 'target', 'symbol', 'exchange']]
        symbols = ['^'+s+'$' for s in df.symbol.tolist()]
        symbols = '|'.join(symbols)
        try:
            res_file = pd.read_csv(self._rpath+'tag_cat_results.csv', sep='\t')
        except FileNotFoundError:
            results = self.query_loop(df, 'standard')
            results = self.clean_results(results, 'standard')
            print("Size of results after cleaning: {}".format(results.shape[0]))
            results.to_csv(self._rpath+'tag_cat_results.csv', sep="\t", index=False)
            res_file = results
        diff = df[~df['symbol'].isin(res_file['symbol'].tolist())].copy()
        try:
            by_name_file = pd.read_csv(self._rpath+'tag_cat_by_name_temp.csv', sep='\t')
            diff = df[~df['symbol'].isin(by_name_file['symbol'].tolist())].copy()
            #results = self.clean_results(by_name_file, 'by_name', diff)
        except FileNotFoundError:
            print("Remaining symbols: {}".format(diff.shape[0]))
            results = self.query_loop(diff, 'by_name')
            results = self.clean_results(results, 'standard')
            results.to_csv(self._rpath+'tag_cat_by_name_temp.csv', sep='\t', index=False)

if __name__ == "__main__":
    queryer = Queryer()
    queryer.run()