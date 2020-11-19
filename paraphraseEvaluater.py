import json
import parserExtractor
import regressionInfer

def extractJson(contentList, dataFilename, outputJsonFilename):
    totalData = {}
    with open(dataFilename, 'r') as dataFile:
        for line in dataFile:
            data = json.loads(line.strip())
            text = data['content']
            totalData[text] = data

    with open(outputJsonFilename, 'w') as outputFile:
        for index, content in enumerate(contentList):
            for text in totalData:
                if content.lower() in text.lower():
                    outputFile.write(json.dumps(totalData[text])+'\n')
                    print(index)
    return None


def generateParaphraseJson(dataFilename, paraphraseList, outputFilename, multiplier=5):
    outputFile = open(outputFilename, 'w')
    with open(dataFilename, 'r') as dataFile:
        paraphraseIndex = 0
        for index, line in enumerate(dataFile):
            data = json.loads(line.strip())
            for i in range(multiplier):
                data['content'] = paraphraseList[paraphraseIndex]
                outputFile.write(json.dumps(data) + '\n')
                paraphraseIndex += 1
    outputFile.close()
    return None


if __name__ == '__main__':
    '''
    contentList = []
    with open('dataset/commTweets/testData_comm', 'r') as listFile:
        for line in listFile:
            contentList.append(line.strip())
    extractJson(contentList, 'dataset/commTweets/features.json', 'dataset/commTweets/test.json')

    paraphraseList = []
    with open('dataset/commTweets/paraphrase.txt', 'r') as listFile:
        for line in listFile:
            paraphraseList.append(line.strip())
    generateParaphraseJson('dataset/commTweets/test.json', paraphraseList, 'dataset/commTweets/paraphrase.json', multiplier=5)
    '''

    #parserExtractor.extractor('dataset/commTweets/paraphrase.txt.predict', 'dataset/commTweets/paraphrase.json', 'dataset/commTweets/paraphrase_feature.json')

    regressionInfer.predictLabel('dataset/commTweets/model/regression_1_0.model', 'dataset/commTweets/paraphrase_feature.json', 'dataset/commTweets/model/tokenizer.pkl', 1)



