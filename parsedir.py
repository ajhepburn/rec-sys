from os import listdir
from os.path import isfile, join, exists
import json, re

# path = "/media/ntfs/st_data/week/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
# data = []

class ParseDir:
    def __init__ (self, directory):
        self.path_data = './data/'
        self.dir = directory
        self.files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]

    """ get_user_tweets is responsible for fetching all of the tweets 
        from disk and putting them in the ./data/ directory.
    """

    def get_user_tweets(self):
        temp_store = {}
        
        for filename in self.files:
            if not exists(self.path_data+filename):
                with open(self.dir+filename) as inputFile:
                        for line in inputFile:
                            data = json.loads(line)
                            user = data['data']['user']['username']
                            tweet_id = data['data']['id']
                            tweet_body = data['data']['body']
                            if user not in data:
                                #store[user] = [tweet]
                                temp_store[user] = [{'id':tweet_id, 'body':tweet_body}]
                            else:
                                temp_store[user].append({'id':tweet_id, 'body':tweet_body})
                with open(self.path_data+filename+".txt", 'w') as fp:
                    for k, v in temp_store.items():
                        json.dump({k:v}, fp)
                        fp.write("\n")

    """ The following functions are responsible for combining the files and their tweets.

    """

    def date_prompt(self, dates):
        regexp = re.compile(r'\d{4}_\d{2}_\d{2}')

        print("\n"+"Please enter a FROM and TO date in the same format (ie. 2018_01_01):")
        date_from = input("FROM> ")
        date_to = input("TO> ")
        
        if date_from in dates and date_to in dates and regexp.match(date_to) and regexp.match(date_from):
            if date_to > date_from:
                return date_from, date_to
            else:
                print("TO date is earlier than FROM date.")
                return 0
        else:
            print("Dates do not exist in store, please try again.")
            return 0

    def combine_structures(self, store, date_from, date_to):
        combined_store = {}
        if exists('./data/'+store[date_from]) and exists('./data/'+store[date_to]):
            for f in store:
                with open('./data/'+f) as inputFile:
                    for line in inputFile:
                        user = list(json.loads(line).keys())[0]
                        tweets = list(json.loads(line).values())[0]
                        if user not in combined_store:
                            combined_store[user] = tweets
                        else:
                            combined_store[user] += tweets

        #combined_store = {k: set(v) for k, v in combined_store.items()}
        for user, tweets in combined_store.items():
            combined_store[user] = {d['body']: d for d in reversed(tweets)}.values()
        return combined_store

    def write_combined_file(self, c_store, file_from, file_to):
        with open('./data/st_comb_'+file_from[-14:-4]+"-"+file_to[-14:-4]+".txt", 'w') as fp:
            for k, v in c_store.items():
                json.dump({k:list(v)}, fp)
                fp.write("\n")

    def combine_files(self):
        title = "COMBINE FILES"
        store = [f for f in listdir('./data/') if isfile(join('./data/', f)) and "stocktwits_messages_" in f]
        store.sort()
        dates = []

        for filename in store:
            dates.append(filename[-14:-4])
        
        print("\n"+title+"\n"+("-"*len(title))+"\n"+"Data is available for the following dates:")
        print("{}, {}".format(", ".join(dates[:-1]), dates[-1]))

        date_from, date_to = None, None
        while date_from == None and date_to == None:
            try:
                date_from, date_to = ParseDir.date_prompt(self, dates)
            except TypeError: pass
        
        date_from_index, date_to_index = [i for i, s in enumerate(store) if date_from in s], [i for i, s in enumerate(store) if date_to in s]
        if date_from_index != None and date_to_index != None:
            c_store = ParseDir.combine_structures(self, store, date_from_index[0], date_to_index[0])
            ParseDir.write_combined_file(self, c_store, store[date_from_index[0]], store[date_to_index[0]])


    



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

pd = ParseDir("/media/ntfs/st_data/month/")
#pd.get_user_tweets()
#pd.combine_files()