from os import listdir
from os.path import isfile, join, exists
from pathlib import Path
from datetime import timedelta, datetime
import time

from utils import lines_that_contain, check_file, date_prompt

import spacy, json, re, itertools, multiprocessing
import more_itertools as mit


# path = "/media/ntfs/st_data/week/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
# data = []

class Parser:
    def __init__ (self, directory):
        self.path_data = './data/'
        self.tweets_path = self.path_data+'tweets/'
        self.dir = directory
        self.files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.files.sort()
        with open('./utilities/slang.txt') as sl:
            slang_terms = json.loads(sl.readline())
            self.slang_terms = [t.lower() for t in slang_terms]
        self.nlp = spacy.load('en')


    def split_files(self, no_of_processes):
        files = [list(c) for c in mit.divide(no_of_processes, self.files)]
        return files


    def tokenise(self, tweet):
        tweet_ct = re.sub(r'\$(\w+)',r'ZZZCASHTAGZZZ\1',tweet)
        tweet_ct = re.sub(r'&#39;s', r"", tweet_ct)
        tweet_ct = re.sub(r'&#39;', r"'", tweet_ct)
        tokens = self.nlp(tweet_ct.lower(), disable=['parser', 'tagger', 'ner'])
        tokens = [token for token in tokens if not token.orth_.isspace() and token.is_alpha and not token.is_stop and token.orth_ not in self.slang_terms and token.lemma_ != '-PRON-' and len(token.orth_) > 3]
        l_tokens = []
        for token in tokens:
            if token.orth_.startswith('zzzcashtagzzz'):
                ct = token.text.replace(u'zzzcashtagzzz','$')
                l_tokens.append(ct)
            else:
                l_token = token.lemma_
                l_tokens.append(l_token)
        tokens = l_tokens
        if len(tokens) > 4:
            return tokens

    def parse_file(self, filename):
        with open(self.tweets_path+filename+".txt", 'w') as fp:
            with open(self.dir+filename) as inputFile:
                for line in inputFile:
                    data = json.loads(line)
                    user = data['data']['user']['username']
                    tweet_id = data['data']['id']
                    tweet_body = data['data']['body']
                    tokens = Parser.tokenise(self, tweet_body)
                    if tokens:
                        entry = {'id':tweet_id, 'body':tweet_body, 'tokens':tokens}
                        json.dump({user:entry}, fp)
                        fp.write("\n")
        

    def get_user_tweets(self, files):
        for filename in files:
            if not exists(self.tweets_path+filename+'.txt'):
                Parser.parse_file(self, filename)

    """ The following functions are responsible for combining the files and their tweets.

    """

    def write_users_file(self):
        title = "WRITE USERS FILE"
        store = [f for f in listdir(self.tweets_path) if isfile(join(self.tweets_path, f)) and "stocktwits_messages_" in f]
        store.sort()
        dates = []

        for filename in store:
            dates.append(filename[-14:-4])
        
        print("\n"+title+"\n"+("-"*len(title))+"\n"+"Data is available for the following dates:")
        print("{}, {}".format(", ".join(dates[:-1]), dates[-1]))

        date_from, date_to = None, None
        while date_from == None and date_to == None:
            try:
                date_from, date_to = date_prompt(self, dates)
            except TypeError: pass
        
        date_from_index, date_to_index = [i for i, s in enumerate(store) if date_from in s], [i for i, s in enumerate(store) if date_to in s]
        if date_from_index != None and date_to_index != None:
            files_selected = store[date_from_index[0]:date_to_index[0]+1]
            users_file = self.path_data+'st_comb_'+files_selected[0][-14:-4]+"-"+files_selected[-1][-14:-4]+"_USERS.txt"
            with open(users_file, 'w') as fp:
                for f in files_selected:
                    print("Collecting users... Current file: {0}".format(f), end="\r")
                    with open(self.tweets_path+f) as inputFile:
                        for line in inputFile:
                            user = list(json.loads(line).keys())[0]
                            fp.write(user+"\n")
            print("\nUsers written to file:", users_file)
            print("Please use command (on Unix filesystems): 'sort -u -o "+users_file+" "+users_file+"' to get unique users")


if __name__ == "__main__":
    pd = Parser("/media/ntfs/st_2017/")
    print("Parsing/Tokenisation began:", str(datetime.now()))
    start_time = time.monotonic()
    
    no_of_processes = multiprocessing.cpu_count()/2
    procs = []
    
    files = pd.split_files(no_of_processes)

    for index, file_list in enumerate(files):
        proc = multiprocessing.Process(target=pd.get_user_tweets, args=(file_list,))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()

    end_time = time.monotonic()
    print("Parsing/Tokenisation Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))