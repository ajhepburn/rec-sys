tweet_bodys = []

with open("data/stocktwits_messages_2018_01_01.json") as f:
    for line in f:
        if line.find("body") == 62:
            tweet_bodys.append(line[68:].split(',"created_at"')[0])