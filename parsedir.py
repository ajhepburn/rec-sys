from os import listdir
from os.path import isfile, join, exists
import json

path = "/media/ntfs/st_data/day/"
files = [f for f in listdir(path) if isfile(join(path, f))]
data = []

def get_tweet_bodies_from_dir():
    for filename in files:
        if not exists('./data/'+filename+".txt"):
            with open(path+filename) as inputFile:
                with open('./data/'+filename+".txt", 'w') as outputFile:
                    for line in inputFile:
                        if line.find('"body":"'):
                            outputFile.write(line[68:].split(',"created_at"')[0]+'\n')

def get_user_tweets():
    data = {}

    for filename in files:
        if not exists('./data/'+filename+".txt"):
            with open(path+filename) as inputFile:
                    for line in inputFile:
                        user = json.loads(line)['data']['user']['username']
                        tweet = json.loads(line)['data']['body']
                        if user not in data:
                            data[user] = [tweet]
                        else:
                            data[user].append(tweet)
            with open('./data/'+filename+".txt", 'w') as fp:
                json.dump(data, fp)

get_user_tweets()