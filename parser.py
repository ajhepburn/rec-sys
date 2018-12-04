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
        self.raw_path = self.path_data+'raw/'
        self.dir = directory
        self.files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.files.sort()
        with open('./utilities/slang.txt') as sl:
            slang_terms = json.loads(sl.readline())
            self.slang_terms = [t.lower() for t in slang_terms]
        self.nlp = spacy.load('en')

    def split_files(self, no_of_processes):
        #dates = [x[-10:] for x in self.files]
        months = list(set([x[-5:-3] for x in self.files]))
        months.sort()
        split_list = [list(c) for c in mit.divide(no_of_processes, months[4:])]

        split_files = []
        for process_allocation in split_list:
            process_data = []
            for month in process_allocation:
                process_data.append([s for s in self.files if month in s[-5:-3]])
            process_data = list(itertools.chain.from_iterable(process_data))
            split_files.append(process_data)
        
        return split_files

            

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
        #tokens = [x.replace(u'zzzcashtagzzz','$') for x in [token.text for token in iter(tokens)]]
        #tokens = [token.lemma_ for token in tokens if not token.startswith('$')]
        tokens = l_tokens
        if len(tokens) > 3:
            return tokens
        else:
            return []

    def parse_file(self, filename):
        with open(self.raw_path+filename+".txt", 'w') as fp:
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
            if not exists(self.raw_path+filename+'.txt'):
                Parser.parse_file(self, filename)

    """ The following functions are responsible for combining the files and their tweets.

    """

    def write_users_file(self):
        title = "WRITE USERS FILE"
        store = [f for f in listdir(self.raw_path) if isfile(join(self.raw_path, f)) and "stocktwits_messages_" in f]
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
                    with open(self.raw_path+f) as inputFile:
                        for line in inputFile:
                            user = list(json.loads(line).keys())[0]
                            fp.write(user+"\n")
            print("\nUsers written to file:", users_file)
            print("Please use command (on Unix filesystems): 'sort -u -o "+users_file+" "+users_file+"' to get unique users")

    # def split_users(self, filename):
    #     users = 
    #     check_file(self.path_data, filename)
    #     with open(self.path_data+filename) as users_file:
    #         f_char = users_file.readline()[0]
    #         for user in users_file:
    #             if 


    # def generate_combined_files(self, filename):
    #     check_file(self.path_data, filename)
    #     num_lines = sum(1 for line in open(self.path_data+filename))
    #     dates = (filename[-31:-21], filename[-20:-10])
    #     files_indexes = ([self.files.index(s) for s in self.files if dates[0] in s][0], [self.files.index(s) for s in self.files if dates[1] in s][0])
    #     files_selected = self.files[files_indexes[0]:files_indexes[1]+1]
        
    #     print("Writing combined file began:", str(datetime.now()))
    #     start_time = time.monotonic()
    #     with open(self.path_data+"st_comb_"+dates[0]+"-"+dates[1]+".txt", "w") as fp:
    #         with open(self.path_data+filename) as users_file:
    #             user_count = 0
    #             for user in users_file:
    #                 user_count += 1
    #                 print("Writing combined file... User: {0}/{1}".format(user_count, num_lines), end="\r")
    #                 username = user.rstrip("\n")
    #                 user_data = {username:[]}
    #                 for f in files_selected:
    #                     with open(self.raw_path+f+".txt") as selected:
    #                         for line in selected:
    #                             if '{"'+username+'":' in line:
    #                                 tweets = list(json.loads(line).values())[0]
    #                                 user_data[username] += tweets
    #                 json.dump(user_data, fp)
    #                 fp.write("\n")
    #     end_time = time.monotonic()
    #     print("Writing Combined File Ended:", str(datetime.now())+".", "Time taken:", timedelta(seconds=end_time - start_time))

                                

            


    # def combine_structures(self, store, date_from, date_to):
    #     combined_store = {}
    #     if exists(self.raw_path+store[date_from]) and exists(self.raw_path+store[date_to]):
    #         for f in store:
    #             print("Combining files... (Current file: {0})".format(f), end="\r")
    #             with open(self.raw_path+f) as inputFile:
    #                 for line in inputFile:
    #                     user = list(json.loads(line).keys())[0]
    #                     tweets = list(json.loads(line).values())[0]
    #                     if user not in combined_store:
    #                         combined_store[user] = tweets
    #                     else:
    #                         combined_store[user] += tweets

    #     #combined_store = {k: set(v) for k, v in combined_store.items()}
    #     for user, tweets in combined_store.items():
    #         combined_store[user] = {d['body']: d for d in reversed(tweets)}.values()
    #     return combined_store

    # def write_combined_file(self, c_store, file_from, file_to):
    #     with open(self.path_data+'st_comb_'+file_from[-14:-4]+"-"+file_to[-14:-4]+".txt", 'w') as fp:
    #         for k, v in c_store.items():
    #             json.dump({k:list(v)}, fp)
    #             fp.write("\n")

    # def combine_files(self):
        
    #         #c_store = Parser.combine_structures(self, store, date_from_index[0], date_to_index[0])
    #         #Parser.write_combined_file(self, c_store, store[date_from_index[0]], store[date_to_index[0]])


    



# def get_tweet_bodies_from_dir():
#     for filename in files:
#         if not exists('./data/'+filename+".txt"):
#             with open(path+filename) as inputFile:
#                 with open('./data/'+filename+".txt", 'w') as outputFile:
#                     for line in inputFile:
#                         if line.find('"body":"'):
#                             outputFile.write(line[68:].split(',"created_at"')[0]+'\n')

# def get_user_tweets():
#     store = {}

#     for filename in files:
#         if not exists('./data/'+filename+".txt"):
#             with open(path+filename) as inputFile:
#                     for line in inputFile:
#                         data = json.loads(line)
#                         user = data['data']['user']['username']
#                         tweet_id = data['data']['id']
#                         tweet_body = data['data']['body']
#                         if user not in data:
#                             #store[user] = [tweet]
#                             store[user] = [{'id':tweet_id, 'body':tweet_body}]
#                         else:
#                             store[user].append({'id':tweet_id, 'body':tweet_body})
#             with open('./data/'+filename+".txt", 'w') as fp:
#                 for k, v in store.items():
#                     json.dump({k:v}, fp)
#                     fp.write("\n")

# def combine_files():
#     def date_prompt():
#         regexp = re.compile(r'\d{4}_\d{2}_\d{2}')

#         print("\n"+"Please enter a FROM and TO date in the same format (ie. 2018_01_01):")
#         date_from = input("FROM> ")
#         date_to = input("TO> ")
        
#         if date_from in dates and date_to in dates and regexp.match(date_to) and regexp.match(date_from):
#             if date_to > date_from:
#                 return date_from, date_to
#             else:
#                 print("TO date is earlier than FROM date.")
#                 return 0
#         else:
#             print("Dates do not exist in store, please try again.")
#             return 0

#     def combine_structures(store, date_from, date_to):
#         combined_store = {}
#         if exists('./data/'+store[date_from]) and exists('./data/'+store[date_to]):
#             for f in store:
#                 with open('./data/'+f) as inputFile:
#                     for line in inputFile:
#                         user = list(json.loads(line).keys())[0]
#                         tweets = list(json.loads(line).values())[0]
#                         if user not in combined_store:
#                             combined_store[user] = tweets
#                         else:
#                             combined_store[user] += tweets

#         #combined_store = {k: set(v) for k, v in combined_store.items()}
#         for user, tweets in combined_store.items():
#             combined_store[user] = {d['body']: d for d in reversed(tweets)}.values()
#         return combined_store

#     def write_combined_file(c_store, file_from, file_to):
#         with open('./data/st_comb_'+file_from[-14:-4]+"-"+file_to[-14:-4]+".txt", 'w') as fp:
#             for k, v in c_store.items():
#                 json.dump({k:list(v)}, fp)
#                 fp.write("\n")


#     title = "COMBINE FILES"
#     store = [f for f in listdir('./data/') if isfile(join('./data/', f)) and "stocktwits_messages_" in f]
#     store.sort()
#     dates = []

#     for filename in store:
#         dates.append(filename[-14:-4])
    
#     print("\n"+title+"\n"+("-"*len(title))+"\n"+"Data is available for the following dates:")
#     print("{}, {}".format(", ".join(dates[:-1]), dates[-1]))

#     date_from, date_to = None, None
#     while date_from == None and date_to == None:
#         try:
#             date_from, date_to = date_prompt()
#         except TypeError:
#             pass
    
#     date_from_index, date_to_index = [i for i, s in enumerate(store) if date_from in s], [i for i, s in enumerate(store) if date_to in s]
#     if date_from_index != None and date_to_index != None:
#         c_store = combine_structures(store, date_from_index[0], date_to_index[0])
#         write_combined_file(c_store, store[date_from_index[0]], store[date_to_index[0]])


#get_user_tweets()
#combine_files()

#pd = Parser("/media/ntfs/st_data/month/")
#pd.get_user_tweets()
#pd.combine_files()



#files = pd.split_files(4)
#pd.get_user_tweets()
#pd.write_users_file()
#pd.generate_combined_files('st_comb_2017_01_01-2017_12_31_USERS.txt')

if __name__ == "__main__":
    pd = Parser("/media/ntfs/st_2017/")
    print("Parsing/Tokenisation began:", str(datetime.now()))
    start_time = time.monotonic()
    
    no_of_processes = 4
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