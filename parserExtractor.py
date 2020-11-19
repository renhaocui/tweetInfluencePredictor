import json

def longestLength(input):
    outputLength = 0
    for key, value in input.items():
        length = 0
        if value != '-1' and value != '_':
            length += 1
            if value == '0':
                if length > outputLength:
                    outputLength = length
                continue
            nextNode = value
            while nextNode != '-1' and nextNode != '_' and nextNode != '0':
                length += 1
                nextNode = input[nextNode]
        if length > outputLength:
            outputLength = length
    return outputLength


def outputHeads(input):
    output = ''
    for key, value in input.items():
        if value[1] == 0:
            output += value[0] + '/' + value[2] + ' '
    return output.strip()


def extractor(parserFilename, dataFilename, outputFilename):
    parserResultFile = open(parserFilename, 'r')
    dataFile = open(dataFilename, 'r')
    #lengthFile = open('dataset/commTweets/parser/clean.length', 'w')
    #headCountFile = open('dataset/commTweets/parser/clean.headCount', 'w')
    #POSCountFile = open('dataset/commTweets/parser/clean.posCount', 'w')
    #POSFile = open('dataset/commTweets/parser/clean.pos', 'w')

    tempData = {}
    tempOutput = {}
    posCount = {'N': 0, 'V': 0, 'A': 0}
    wordList = []
    posList = []
    IDData = []
    totalData = []
    for line in dataFile:
        data = json.loads(line.strip())
        IDData.append(data['id'])
        totalData.append(data)
    dataFile.close()

    lengthList = []
    headCountList = []
    posCountList = []
    index = 0
    for line in parserResultFile:
        if line.strip() != '':
            items = line.strip().split()
            tempData[items[0]] = items[6]
            tempOutput[int(items[0])] = (items[1], int(items[6]), items[4])
            wordList.append(items[1])
            posList.append(items[4])
            if items[4] in ['N', '^']:
                posCount['N'] += 1
            elif items[4] == 'V':
                posCount['V'] += 1
            elif items[4] in ['A', 'R']:
                posCount['A'] += 1
        else:
            #longLen = longestLength(tempData)
            #lengthFile.write(str(longLen) + ' :: ' + str(IDData[index]) + '\n')
            #headCountFile.write(str(len(outputHeads(tempOutput).split())) + ' :: ' + str(IDData[index]) + '\n')
            #POSCountFile.write(str(posCount['N']) + ' ' + str(posCount['V']) + ' ' + str(posCount['A']) + ' :: ' + str(IDData[index]) + '\n')
            totalData[index]['length'] = longestLength(tempData)
            totalData[index]['head_count'] = len(outputHeads(tempOutput).split())
            totalData[index]['pos_count'] = posCount
            #out = ''
            #for num in range(len(wordList)):
            #    out += wordList[num]+'/'+posList[num]+';'
            #POSFile.write(out[:-1]+' :: '+str(IDData[index]) +'\n')

            tempData = {}
            tempOutput = {}
            wordList = []
            posList = []
            posCount = {'N': 0, 'V': 0, 'A': 0}
            index += 1

    print(len(totalData))
    print(index)

    parserResultFile.close()
    #lengthFile.close()
    #headCountFile.close()
    #POSCountFile.close()
    #POSFile.close()

    outputFile = open(outputFilename, 'w')
    for data in totalData:
        outputFile.write(json.dumps(data)+'\n')
    outputFile.close()


def extractor2():
    inputFile = open('dataset/experiment/content/total.content.predict', 'r')
    idFile = open('dataset/experiment/content/total.id', 'r')
    lengthFile = open('dataset/experiment/parser/total.length', 'w')
    headCountFile = open('dataset/experiment/parser/total.headCount', 'w')
    POSCountFile = open('dataset/experiment/parser/total.posCount', 'w')
    POSFile = open('dataset/experiment/parser/total.pos', 'w')

    tempData = {}
    tempOutput = {}
    posCount = {'N': 0, 'V': 0, 'A': 0}
    wordList = []
    posList = []
    IDData = []
    for line in idFile:
        IDData.append(line.strip())
    idFile.close()

    index = 0
    for line in inputFile:
        if line.strip() != '':
            items = line.strip().split()
            tempData[items[0]] = items[6]
            tempOutput[int(items[0])] = (items[1], int(items[6]), items[4])
            wordList.append(items[1])
            posList.append(items[4])
            if items[4] in ['N', '^']:
                posCount['N'] += 1
            elif items[4] == 'V':
                posCount['V'] += 1
            elif items[4] in ['A', 'R']:
                posCount['A'] += 1
        else:
            longLen = longestLength(tempData)
            lengthFile.write(str(longLen) + ' :: ' + str(IDData[index]) + '\n')
            headCountFile.write(str(len(outputHeads(tempOutput).split())) + ' :: ' + str(IDData[index]) + '\n')
            POSCountFile.write(str(posCount['N']) + ' ' + str(posCount['V']) + ' ' + str(posCount['A']) + ' :: ' + str(IDData[index]) + '\n')
            out = ''
            for num in range(len(wordList)):
                out += wordList[num]+'/'+posList[num]+';'
            POSFile.write(out[:-1]+' :: '+str(IDData[index]) +'\n')

            tempData = {}
            tempOutput = {}
            wordList = []
            posList = []
            posCount = {'N': 0, 'V': 0, 'A': 0}
            index += 1


    inputFile.close()
    lengthFile.close()
    headCountFile.close()
    POSCountFile.close()
    POSFile.close()


def extractKey():
    POSFile = open('dataset/experiment/parser/total.pos', 'r')
    keyFile = open('dataset/experiment/parser/total.key', 'w')
    for line in POSFile:
        out = ''
        temp = line.strip().split(' :: ')
        words = temp[0].split(';')
        for item in words:
            seg = item.split('/')
            if len(seg) > 1 and seg[1] in ['N', 'M', 'Z', 'S', 'L', '^', 'V', 'A', 'R']:
                out += seg[0] + ' '
        keyFile.write(out.strip()+' :: '+temp[1]+'\n')
    POSFile.close()
    keyFile.close()


if __name__ == '__main__':
    extractor('dataset/commTweets/test.content.predict', 'dataset/commTweets/test.json', 'dataset/commTweets/test_features.json')
    #extractKey()