__author__ = 'rencui'
import json

def contenterExtractor():
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    outputContentFile = open('dataset/experiment/content/clean.content', 'w')

    for line in dataFile:
        data = json.loads(line.strip())
        outputContentFile.write(data['content'].encode('utf-8')+'\n')

    dataFile.close()
    outputContentFile.close()


if __name__ == "__main__":
    contenterExtractor()