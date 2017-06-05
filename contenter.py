__author__ = 'rencui'
import json
import tweetTextCleaner

def contenterExtractor():
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    outputContentFile = open('dataset/experiment/content/clean.content', 'w')

    for line in dataFile:
        data = json.loads(line.strip())
        outputContentFile.write(data['content'].encode('utf-8')+'\n')

    dataFile.close()
    outputContentFile.close()


def generateTotalContent():
    inputFile = open('dataset/experiment/total.json', 'r')
    contentFile = open('dataset/experiment/content/total.content', 'w')
    matchIDFile = open('dataset/experiment/content/total.id', 'w')
    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        content = tweetTextCleaner.tweetCleaner(text)
        contentFile.write(content+'\n')
        matchIDFile.write(str(tweetID)+'\n')
    inputFile.close()
    contentFile.close()
    matchIDFile.close()


def splitContent():
    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    dataFile = open('dataset/experiment/clean.labeled', 'r')
    outputContentTrainFile = open('dataset/experiment/content/clean.content.train', 'w')
    outputContentTestFile = open('dataset/experiment/content/clean.content.test', 'w')
    trainData = []
    testData = []
    totalData = []

    for line in dataFile:
        temp = json.loads(line.strip())
        content = temp['content'].encode('utf-8').replace('"', "'")
        if str(temp['id']) in trainIDs:
            outputContentTrainFile.write(content+'\n')
            trainData.append({'tweet2': content, 'tweet1': content})
        elif str(temp['id']) in testIDs:
            outputContentTestFile.write(content + '\n')
            testData.append(content)
        totalData.append(content)

    dataFile.close()
    outputContentTrainFile.close()
    outputContentTestFile.close()

    print len(trainData)
    print len(testData)

    trainJsonFile = open('dataset/experiment/content/train.json', 'w')
    testJsonFile = open('dataset/experiment/content/test.json', 'w')
    totalJsonFile = open('dataset/experiment/content/total.json', 'w')
    trainJsonFile.write(json.dumps(trainData))
    testJsonFile.write(json.dumps(testData))
    totalJsonFile.write(json.dumps(totalData))
    trainJsonFile.close()
    testJsonFile.close()
    totalJsonFile.close()


def generateContent_MIT(split=False):
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    outputContentFile = open('dataset/experiment/content/clean.MIT', 'w')

    outputData = []
    contentData = []
    for line in dataFile:
        data = json.loads(line.strip())
        content = data['content'].encode('utf-8')
        out = {'tweet1': content, 'tweet2': content}
        outputData.append(out)
        contentData.append(content)
    outputContentFile.write(json.dumps(outputData))



    dataFile.close()
    outputContentFile.close()

if __name__ == "__main__":
    #contenterExtractor()
    #splitContent()
    #generateContent_MIT(True)
    generateTotalContent()