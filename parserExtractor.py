__author__ = 'renhao.cui'
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


def extractor():
    inputFile = open('dataset/experiment/content/clean.content.predict', 'r')
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    lengthFile = open('dataset/experiment/parser/clean.length', 'w')
    headCountFile = open('dataset/experiment/parser/clean.headCount', 'w')
    POSCountFile = open('dataset/experiment/parser/clean.posCount', 'w')

    tempData = {}
    tempOutput = {}
    posCount = {'N': 0, 'V': 0, 'A': 0}
    cleanData = []
    for line in dataFile:
        data = json.loads(line.strip())
        cleanData.append(data['id'])
    dataFile.close()

    index = 0
    for line in inputFile:
        if line.strip() != '':
            words = line.strip().split()
            tempData[words[0]] = words[6]
            tempOutput[int(words[0])] = (words[1], int(words[6]), words[4])
            if words[4] in ['N', '^']:
                posCount['N'] += 1
            elif words[4] == 'V':
                posCount['V'] += 1
            elif words[4] in ['A', 'R']:
                posCount['A'] += 1
        else:
            longLen = longestLength(tempData)
            lengthFile.write(str(longLen) + ' :: ' + str(cleanData[index]) + '\n')
            headCountFile.write(str(len(outputHeads(tempOutput).split())) + ' :: ' + str(cleanData[index]) + '\n')
            POSCountFile.write(str(posCount['N']) + ' ' + str(posCount['V']) + ' ' + str(posCount['A']) + ' :: ' + str(cleanData[index]) + '\n')
            tempData = {}
            tempOutput = {}
            posCount = {'N': 0, 'V': 0, 'A': 0}
            index += 1


    inputFile.close()
    lengthFile.close()
    headCountFile.close()
    POSCountFile.close()


if __name__ == '__main__':
    extractor()
