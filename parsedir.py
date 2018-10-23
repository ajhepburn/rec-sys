from os import listdir
from os.path import isfile, join, exists

path = "/media/ntfs/st_data/"
files = [f for f in listdir(path) if isfile(join(path, f))]
data = []

for filename in files:
    if not exists('./data/'+filename+".txt"):
        with open(path+filename) as inputFile:
            with open('./data/'+filename+".txt", 'w') as outputFile:
                for line in inputFile:
                    if line.find('"body":"'):
                        outputFile.write(line[68:].split(',"created_at"')[0]+'\n')
        