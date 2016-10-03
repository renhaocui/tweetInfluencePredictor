import subprocess
import operator
from sklearn.feature_extraction.text import *
from sklearn import cluster
import shutil
import json

__author__ = 'rencui'


def analysisByDay():
    posDayCount = {}
    negDayCount = {}
    posFile = open('dataset/experiment/ranked/total.pos', 'r')
    negFile = open('dataset/experiment/ranked/total.neg', 'r')
    for line in posFile:
        data = line.strip().split(' : ')
        day = data[1]
        if day not in posDayCount:
            posDayCount[day] = 1.0
        else:
            posDayCount[day] += 1.0
    posFile.close()

    for line in negFile:
        data = line.strip().split(' : ')
        day = data[1]
        if day not in negDayCount:
            negDayCount[day] = 1.0
        else:
            negDayCount[day] += 1.0
    negFile.close()

    print posDayCount
    print negDayCount


def maxIndex(input, num):
    line = {}
    for index in range(len(input)):
        line[index] = float(input[index])
    sorted_line = sorted(line.iteritems(), key=operator.itemgetter(1), reverse=True)
    output = []
    for i in range(num):
        output.append(sorted_line[i][0])
    return output


def fileLineCount(fileList):
    outputList = []
    for fileName in fileList:
        with open(fileName) as f:
            outputList.append(sum(1 for _ in f))
    return outputList


def brandGrouper(groupTitle, groupSize):
    print 'brand grouping...'
    brandMapper = {}
    idMapper = {}
    groupFile = open(groupTitle + '.list', 'r')
    for index, line in enumerate(groupFile):
        brands = line.strip().split()
        for brand in brands:
            brandMapper[brand] = index
    groupFile.close()

    inputFile = open('dataset/experiment/clean.labeled', 'r')
    lengthFile = open('dataset/experiment/parser/clean.length', 'r')
    headCountFile = open('dataset/experiment/parser/clean.headCount', 'r')
    POSCountFile = open('dataset/experiment/parser/clean.posCount', 'r')

    outputFileList = []
    lengthFileList = []
    headFileList = []
    POSCountFileList = []
    countFileList = []

    for index in range(groupSize):
        tempFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.labeled', 'w')
        countFileList.append('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.labeled')
        outputFileList.append(tempFile)
        tempLengthFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.length', 'w')
        lengthFileList.append(tempLengthFile)
        tempHeadFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.headCount', 'w')
        headFileList.append(tempHeadFile)
        tempPOSCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.posCount', 'w')
        POSCountFileList.append(tempPOSCountFile)

    for line in inputFile:
        item = json.loads(line.strip())
        brand = item['brand']
        if brand in brandMapper:
            outputFileList[brandMapper[brand]].write(line)
        idMapper[item['id']] = brand

    for line in lengthFile:
        items = line.strip().split(' :: ')
        if items[1] in brandMapper:
            lengthFileList[brandMapper[items[1]]].write(line)

    for line in headCountFile:
        items = line.strip().split(' :: ')
        if items[1] in brandMapper:
            headFileList[brandMapper[items[1]]].write(line)

    for line in POSCountFile:
        items = line.strip().split(' :: ')
        if items[1] in brandMapper:
            POSCountFileList[brandMapper[items[1]]].write(line)

    for index in range(groupSize):
        outputFileList[index].close()
        lengthFileList[index].close()
        headFileList[index].close()
        POSCountFileList[index].close()
    inputFile.close()
    lengthFile.close()
    headCountFile.close()
    POSCountFile.close()
    print 'Brand Group sizes: ' + str(fileLineCount(countFileList))


def dayGrouper(groupTitle, groupSize):
    print 'day grouping...'
    dayMapper = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

    inputFile = open('dataset/experiment/clean.labeled', 'r')
    lengthFile = open('dataset/experiment/parser/clean.length', 'r')
    headCountFile = open('dataset/experiment/parser/clean.headCount', 'r')
    POSCountFile = open('dataset/experiment/parser/clean.posCount', 'r')

    outputFileList = []
    lengthFileList = []
    headFileList = []
    POSCountFileList = []
    countFileList = []

    for index in range(groupSize):
        tempFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.labeled', 'w')
        countFileList.append('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.labeled')
        outputFileList.append(tempFile)
        tempLengthFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.length', 'w')
        lengthFileList.append(tempLengthFile)
        tempHeadFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.headCount', 'w')
        headFileList.append(tempHeadFile)
        tempPOSCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(index) + '.posCount', 'w')
        POSCountFileList.append(tempPOSCountFile)

    for line in inputFile:
        item = json.loads(line.strip())
        day = item['day']
        if day in dayMapper:
            outputFileList[dayMapper[day]].write(line)

    for line in lengthFile:
        items = line.strip().split(' :: ')
        if items[1] in dayMapper:
            lengthFileList[dayMapper[items[1]]].write(line)

    for line in headCountFile:
        items = line.strip().split(' :: ')
        if items[1] in dayMapper:
            headFileList[dayMapper[items[1]]].write(line)

    for line in POSCountFile:
        items = line.strip().split(' :: ')
        if items[1] in dayMapper:
            POSCountFileList[dayMapper[items[1]]].write(line)

    for index in range(groupSize):
        outputFileList[index].close()
        lengthFileList[index].close()
        headFileList[index].close()
        POSCountFileList[index].close()
    inputFile.close()
    lengthFile.close()
    headCountFile.close()
    print 'Day Group sizes: ' + str(fileLineCount(countFileList))


def topicGrouper(groupSize):
    print 'topic grouping...'
    data = []
    # punctuations = set(string.punctuation)
    inputFile = open('dataset/experiment/clean.labeled', 'r')
    csvFile = open('LDA/LDAinput.csv', 'w')
    listFile = open('LDA/LDAinput.list', 'w')
    for line in inputFile:
        item = json.loads(line.strip())
        data.append(item)
    inputFile.close()

    for tweet in data:
        csvFile.write(tweet['content'].encode('utf-8').replace('"', '\'') + '\n')
        listFile.write(str(tweet['id']) + '\n')
    csvFile.close()
    listFile.close()

    print 'running LDA...'
    subprocess.check_output('java -Xmx1024m -jar LDA/tmt-0.4.0.jar LDA/assign.scala', shell=True)

    distFile = open('LDA/TMTSnapshots/document-topic-distributions.csv', 'r')
    topicOut = {}
    for line in distFile:
        seg = line.strip().split(',')
        if seg[1] != 'NaN':
            topicOutList = maxIndex(seg[1:], 3)
            topicOut[int(seg[0])] = topicOutList
    distFile.close()

    topicMapper = {}
    for index, value in topicOut.items():
        topicMapper[data[index]['id']] = value[0]

    inputFile = open('dataset/experiment/clean.labeled', 'r')
    lengthFile = open('dataset/experiment/parser/clean.length', 'r')
    headCountFile = open('dataset/experiment/parser/clean.headCount', 'r')
    posCountFile = open('dataset/experiment/parser/clean.posCount', 'r')

    outputFileList = []
    lengthFileList = []
    headFileList = []
    posCountFileList = []
    countFileList = []
    for index in range(groupSize):
        tempPosFile = open('dataset/experiment/groups/topicGroup/data_' + str(index) + '.labeled', 'w')
        countFileList.append('dataset/experiment/groups/topicGroup/data_' + str(index) + '.labeled')
        outputFileList.append(tempPosFile)
        tempPosParseLengthFile = open('dataset/experiment/groups/topicGroup/data_' + str(index) + '.length', 'w')
        lengthFileList.append(tempPosParseLengthFile)
        tempPosParseHeadFile = open('dataset/experiment/groups/topicGroup/data_' + str(index) + '.headCount', 'w')
        headFileList.append(tempPosParseHeadFile)
        tempPosPOSCountFile = open('dataset/experiment/groups/topicGroup/data_' + str(index) + '.posCount', 'w')
        posCountFileList.append(tempPosPOSCountFile)

    for line in inputFile:
        item = json.loads(line.strip())
        id = item['id']
        if id in topicMapper:
            outputFileList[topicMapper[id]].write(line)

    for line in lengthFile:
        items = line.strip().split(' :: ')
        if items[1] in topicMapper:
            lengthFileList[topicMapper[items[1]]].write(line)

    for line in headCountFile:
        items = line.strip().split(' :: ')
        if items[1] in topicMapper:
            headFileList[topicMapper[items[1]]].write(line)

    for line in posCountFile:
        items = line.strip().split(' :: ')
        if items[1] in topicMapper:
            posCountFileList[topicMapper[items[1]]].write(line)

    for index in range(groupSize):
        outputFileList[index].close()
        lengthFileList[index].close()
        headFileList[index].close()
        posCountFileList[index].close()
    inputFile.close()
    lengthFile.close()
    headCountFile.close()
    posCountFile.close()
    print 'Topic Group size: ' + str(fileLineCount(countFileList))


def similarityGrouper(groupSize):
    print 'similarity grouping...'
    tweets = []
    ids = []
    idMapper = {}
    inputFile = open('dataset/experiment/clean.labeled', 'r')

    for line in inputFile:
        item = json.loads(line.strip())
        tweets.append(item['content'])
        ids.append(item['id'])
    inputFile.close()

    # vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', binary='True')
    matrix = vectorizer.fit_transform(tweets)
    print len(vectorizer.get_feature_names())

    print 'Running Kmeans...'
    kmeans = cluster.KMeans(n_clusters=groupSize, init='k-means++')
    kmeans.fit(matrix)
    for index, label in enumerate(kmeans.labels_):
        idMapper[ids[index]] = label

    inputFile = open('dataset/experiment/clean.labeled', 'r')
    posParseLengthFile = open('dataset/experiment/parser/clean.length', 'r')
    posHeadCountFile = open('dataset/experiment/parser/clean.headCount', 'r')
    posPOSCountFile = open('dataset/experiment/parser/clean.posCount', 'r')

    outputFileList = []
    lengthFileList = []
    headCountFileList = []
    posCountFileList = []
    # tweetID: [group, brand, performanceScore, day, time, text, parserLength, headCount]
    posDetailData = {}
    countFileList = []
    for index in range(groupSize):
        tempinputFile = open('dataset/experiment/groups/simGroup/data_' + str(index) + '.labeled', 'w')
        countFileList.append('dataset/experiment/groups/simGroup/data_' + str(index) + '.labeled')
        outputFileList.append(tempinputFile)
        tempLengthFile = open('dataset/experiment/groups/simGroup/data_' + str(index) + '.length', 'w')
        lengthFileList.append(tempLengthFile)
        headCountFile = open('dataset/experiment/groups/simGroup/data_' + str(index) + '.headCount', 'w')
        headCountFileList.append(headCountFile)
        tempPOSCountFile = open('dataset/experiment/groups/simGroup/data_' + str(index) + '.posCount', 'w')
        posCountFileList.append(tempPOSCountFile)

    for line in inputFile:
        item = json.loads(line.strip())
        id = item['id']
        if id in idMapper:
            outputFileList[idMapper[id]].write(line)
            # posDetailData[id] = [idMapper[items[5]], items[4], items[0], items[1], items[2], items[3]]

    for line in posParseLengthFile:
        items = line.strip().split(' :: ')
        if items[1] in idMapper:
            lengthFileList[idMapper[items[1]]].write(line)
            # posDetailData[items[1]].append(items[0])

    for line in posHeadCountFile:
        items = line.strip().split(' :: ')
        if items[1] in idMapper:
            headCountFileList[idMapper[items[1]]].write(line)
            # posDetailData[items[1]].append(items[0])

    for line in posPOSCountFile:
        items = line.strip().split(' :: ')
        if items[1] in idMapper:
            posCountFileList[idMapper[items[1]]].write(line)

    for index in range(groupSize):
        outputFileList[index].close()
        lengthFileList[index].close()
        headCountFileList[index].close()
        posCountFileList[index].close()
    inputFile.close()
    posParseLengthFile.close()
    posHeadCountFile.close()
    posPOSCountFile.close()
    print 'Similarity Group sizes: ' + str(fileLineCount(countFileList))

    '''
    posDetailFile = open('adData/analysis/groups/simGroup/details.pos', 'w')
    negDetailFile = open('adData/analysis/groups/simGroup/details.neg', 'w')
    for id, value in posDetailData.items():
        posDetailFile.write(id+'\t'+str(value[0])+'\t'+value[1]+'\t'+value[2]+'\t'+value[3]+'\t'+value[4]+'\t'+value[6]+'\t'+value[7]+'\n')
    for id, value in negDetailData.items():
        negDetailFile.write(id+'\t'+str(value[0])+'\t'+value[1]+'\t'+value[2]+'\t'+value[3]+'\t'+value[4]+'\t'+value[6]+'\t'+value[7]+'\n')
    posDetailFile.close()
    negDetailFile.close()
    '''


def totalGrouper():
    shutil.copy2('dataset/experiment/clean.labeled', 'dataset/experiment/groups/totalGroup/data_0.labeled')
    shutil.copy2('dataset/experiment/parser/clean.headCount', 'dataset/experiment/groups/totalGroup/data_0.headCount')
    shutil.copy2('dataset/experiment/parser/clean.length', 'dataset/experiment/groups/totalGroup/data_0.length')
    shutil.copy2('dataset/experiment/parser/clean.posCount', 'dataset/experiment/groups/totalGroup/data_0.posCount')
    print 'Total Group size: ' + str(fileLineCount(['dataset/experiment/groups/totalGroup/data_0.labeled']))


if __name__ == '__main__':
    # totalGrouper()
    # brandGrouper('brandGroup', 3)
    # brandGrouper('brandGroup', 3)
    # similarityGrouper(5)
    topicGrouper(5)
