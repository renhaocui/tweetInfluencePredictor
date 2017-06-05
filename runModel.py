__author__ = 'rencui'
import json
from afinn import Afinn
import numpy
import utilities
from textstat.textstat import textstat
from sklearn.feature_extraction.text import *
#from nltk.stem.porter import *
from tokenizer import simpleTokenize
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import sys
import operator
from sklearn.model_selection import KFold
import gensim
import word2vecReader

reload(sys)
sys.setdefaultencoding('utf8')

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}

#stemmer = PorterStemmer()
'''
def stemContent(input):
    words = simpleTokenize(input)
    out = ''
    for word in words:
        temp = stemmer.stem(word.encode('utf-8').decode('utf-8'))
        out += temp + ' '
    return out.strip()


def dataSplit(features, labels, seed):
    lineNum = len(labels)
    labels_train = []
    labels_test = []
    features_test = features[seed]
    if seed != 0:
        features_train = features[0]
    else:
        features_train = features[1]

    firstRoundTest = True
    firstRoundTrain = True
    for index in range(lineNum):
        if index % 5 == seed:
            labels_test.append(labels[index])
            if firstRoundTest:
                firstRoundTest = False
            else:
                features_test = hstack((features_test, features[index]), format='csr')
        else:
            labels_train.append(labels[index])
            if firstRoundTrain:
                firstRoundTrain = False
            else:
                features_train = hstack((features_train, features[index]), format='csr')

    return features_train, features_test, labels_train, labels_test


'''
def shuffleTweets(groupSize, groupTitle, labelMode):
    for group in range(groupSize):
        idData = []
        inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.headCount', 'r')
        for line in inputFile:
            temp = line.strip().split(' :: ')
            idData.append(temp[1])
        inputFile.close()
        kf = KFold(n_splits=5, shuffle=True)
        for roundNum, (train_indices, test_indices) in enumerate(kf.split(idData)):
            outputData = {'test': [], 'train': []}
            for index in train_indices:
                outputData['train'].append(idData[index])
            for index in test_indices:
                outputData['test'].append(idData[index])
            if labelMode == 1:
                title = 'totalGroup10'
            if labelMode == 2:
                title = 'totalGroup2'
            outputFile = open('dataset/experiment/groups/' + title + '/indices/' + str(roundNum)+ '.indices', 'w')
            outputFile.write(json.dumps(outputData)+'\n')
            outputFile.close()


def shuffleTweets2(groupSize, groupTitle, removeOutliers=True):
    idData = []
    if removeOutliers:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    else:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '_full.labeled', 'r')
    for line in inputFile:
        temp = json.loads(line.strip())
        idData.append(str(temp['id']))
    inputFile.close()

    kf = KFold(n_splits=5, shuffle=True)
    for roundNum, (train_indices, test_indices) in enumerate(kf.split(idData)):
        outputData = {'test': [], 'train': []}
        for index in train_indices:
            outputData['train'].append(idData[index])
        for index in test_indices:
            outputData['test'].append(idData[index])
        if removeOutliers:
            outputFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '.'+str(roundNum), 'w')
        else:
            outputFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '_full.' + str(roundNum),
                              'w')
        outputFile.write(json.dumps(outputData)+'\n')
        outputFile.close()


def shuffleTweetsSimple(removeOutliers=True):
    idData = []
    if removeOutliers:
        inputFile = open('dataset/experiment/simple_1.labeled', 'r')
    else:
        inputFile = open('dataset/experiment/simple_1_full.labeled', 'r')
    for line in inputFile:
        temp = json.loads(line.strip())
        idData.append(str(temp['id']))
    inputFile.close()

    kf = KFold(n_splits=5, shuffle=True)
    for roundNum, (train_indices, test_indices) in enumerate(kf.split(idData)):
        outputData = {'test': [], 'train': []}
        for index in train_indices:
            outputData['train'].append(idData[index])
        for index in test_indices:
            outputData['test'].append(idData[index])
        if removeOutliers:
            outputFile = open('dataset/experiment/fold_indicies/simple_1.'+str(roundNum), 'w')
        else:
            outputFile = open('dataset/experiment/fold_indicies/simple_1_full.' + str(roundNum),
                              'w')
        outputFile.write(json.dumps(outputData)+'\n')
        outputFile.close()


def writeLabels(groupSize, groupTitle, labelMode):
    for group in range(groupSize):
        inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.labeled', 'r')
        labels = []
        for line in inputFile:
            item = json.loads(line.strip())
            if labelMode == 1:
                labels.append(item['label'])
            else:
                if item['label'] > 5:
                    labels.append(1)
                else:
                    labels.append(0)
        roundNum = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_indices, test_indices in kf.split(labels):
            trainLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.train', 'w')
            testLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.test', 'w')
            trainIndexFile = open('outputs/' + groupTitle + '/Index/' + str(roundNum) + '.train', 'w')
            testIndexFile = open('outputs/' + groupTitle + '/Index/' + str(roundNum) + '.test', 'w')
            for train_index in train_indices:
                trainLabelFile.write(str(labels[train_index])+'\n')
                trainIndexFile.write(str(train_index)+'\n')
            for test_index in test_indices:
                testLabelFile.write(str(labels[test_index]) + '\n')
                testIndexFile.write(str(test_index)+'\n')
            roundNum += 1
            trainLabelFile.close()
            testLabelFile.close()
            trainIndexFile.close()
            testIndexFile.close()


# vectorMode 1: tfidf, 2: binaryCount
# featureMode 0: semantic only, 1: vector only, 2: both
def runModel(groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, recordProb, ablationIndex):
    outputFile = 'results/ablation.result'
    resultFile = open(outputFile, 'a')

    print groupTitle
    resultFile.write(groupTitle + '\n')
    resultFile.write('Label Mode: ' + str(labelMode) + '\n')
    for group in range(groupSize):
        print 'group: ' + str(group)
        resultFile.write('group: ' + str(group) + '\n')
        print 'loading data...'
        errorIDList = []
        ids = []
        contents = []
        #scores = []
        days = []
        time = []
        labels = []
        usernames = []
        additionalFeatures = []
        authorFollowers = []
        authorStatusCount = []
        authorFavoriteCount = []
        authorListedCount = []
        authorIntervals = []

        inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.labeled', 'r')
        for line in inputFile:
            item = json.loads(line.strip())
            ids.append(str(item['id']))
            contents.append(item['content'])
            if labelMode == 1:
                labels.append(item['label'])
            elif labelMode == 2:
                if item['label'] > splitNum:
                    labels.append(1)
                else:
                    labels.append(0)
            if contentFeatures:
                time.append(utilities.hourMapper(item['hour']))
                usernames.append(item['mentions'])
                days.append(dayMapper[item['day']])
                #scores.append(float(item['score']))
                authorFollowers.append(int(item['author_followers_count']))
                authorStatusCount.append(int(item['author_statuses_count']))
                authorFavoriteCount.append(int(item['author_favorite_count']))
                authorListedCount.append(int(item['author_listed_count']))
                authorIntervals.append(int(item['authorInterval']))
        inputFile.close()

        if vectorMode == 1:
            print 'Generating tfidf vectors...'
            resultFile.write('tfidf \n')
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
            vectorMatrix = vectorizer.fit_transform(contents)
        elif vectorMode == 2:
            print 'Generating binary ngram vectors...'
            resultFile.write('binary count \n')
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                         binary='True')
            vectorMatrix = vectorizer.fit_transform(contents)
        else:
            resultFile.write('no vector features \n')

        if embeddingMode == 1:
            embeddingFeature = numpy.load('dataset/experiment/vector/CMU_total_emd.npy')
            resultFile.write('CMU Twitter Hashtag embedding'+'\n')
        elif embeddingMode == 2:
            model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
            tempFeatures = []
            for content in contents:
                words = simpleTokenize(content)
                tempFeatures.append(numpy.array(model.infer_vector(words)))
            embeddingFeature = numpy.array(tempFeatures)
            resultFile.write('MIT Twitter Embedding'+'\n')
        elif embeddingMode == 3:
            w2v = word2vecReader.Word2Vec()
            embModel = w2v.loadModel()
            tempFeatures = []
            for id, content in zip(ids, contents):
                tweetVec = utilities.content2vec(embModel, content)
                if tweetVec == None:
                    errorIDList.append(id)
                    tempFeatures.append(numpy.zeros(400))
                else:
                    tempFeatures.append(tweetVec)
            embeddingFeature = numpy.array(tempFeatures)
            resultFile.write('gneral twitter word2vec + CBOW' + '\n')
        else:
            resultFile.write('no twitter embedding'+'\n')

        if len(embeddingFeature) != len(labels):
            print 'Embedding vector size error...'

        if contentFeatures:
            resultFile.write('with content features'+'\n')
            mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
            afinn = Afinn()

            parseLength = []
            headCount = []
            POScounts = []
            lengthFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.length', 'r')
            headCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.headCount', 'r')
            posCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.posCount', 'r')

            for line in lengthFile:
                parseLength.append(int(line.strip(' :: ')[0]))
            for line in headCountFile:
                headCount.append(int(line.strip(' :: ')[0]))
            for line in posCountFile:
                POScounts.append(utilities.POSRatio(line.strip().split(' :: ')[0].split(' ')))

            headCountFile.close()
            lengthFile.close()
            posCountFile.close()

            for index, content in enumerate(contents):
                temp = []
                words = simpleTokenize(content)
                twLen = float(len(words))
                sentiScore = afinn.score(content)
                readScore = textstat.coleman_liau_index(content)

                if ablationIndex != 0:
                    temp.append(sentiScore / twLen)
                if ablationIndex != 1:
                    temp.append(twLen)
                    temp.append(readScore)
                    temp.append(parseLength[index] / twLen)
                    temp.append(headCount[index] / twLen)
                if ablationIndex != 2:
                    temp.append(authorStatusCount[index]/authorIntervals[index])
                    temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                    temp.append(authorListedCount[index]/authorFollowers[index])
                if ablationIndex != 3:
                    temp.append(days[index])
                    temp.append(time[index])
                if ablationIndex != 4:
                    if any(char.isdigit() for char in content):
                        temp.append(1)
                    else:
                        temp.append(0)
                if ablationIndex != 5:
                    temp += POScounts[index]
                if ablationIndex != 6:
                    # temp.append(content.count('URRL'))
                    if content.count('http://URL') > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                    # temp.append(content.count('HHTTG'))
                    if content.count('#HTG') > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                    # temp.append(content.count('USSERNM'))
                    if content.count('@URNM') > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                if ablationIndex != 7:
                    # temp.append(content.count('!'))
                    if content.count('!') > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                    # temp.append(content.count('?'))
                    if content.count('?') > 0:
                        temp.append(1)
                    else:
                        temp.append(0)
                if ablationIndex != 8:
                    mentionFlag = 0
                    mentionFollowers = 0
                    userCount = 0.0
                    for user in usernames[index]:
                        if user in mentionMapper:
                            userCount += 1
                            if mentionMapper[user][0] == 1:
                                mentionFlag = 1
                            mentionFollowers += mentionMapper[user][1]
                    temp.append(mentionFlag)

                    if userCount == 0:
                        temp.append(0.0)
                    else:
                        temp.append(mentionFollowers / userCount)

                additionalFeatures.append(numpy.array(temp))
        else:
            resultFile.write('no content features'+'\n')

        if featureMode == 0:
            resultFile.write('content features only \n')
            features = csr_matrix(numpy.array(additionalFeatures))
        elif featureMode == 1:
            resultFile.write('vector features only \n')
            features = vectorMatrix
        elif featureMode == 2:
            resultFile.write('content and embedding'+'\n')
            features = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
        elif featureMode == 3:
            resultFile.write('embedding only'+'\n')
            features = csr_matrix(embeddingFeature)
        else:
            resultFile.write('vector and content features \n')
            features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')
        resultFile.flush()

        featureData = {}
        labelData = {}
        for lineNum, label in enumerate(labels):
            labelData[ids[lineNum]] = label
            featureData[ids[lineNum]] = features.getrow(lineNum)

        precisionSumList = {}
        recallSumList = {}
        F1SumList = {}
        aucSumList = {}
        for trainMode in trainModeList:
            precisionSumList[trainMode] = 0.0
            recallSumList[trainMode] = 0.0
            F1SumList[trainMode] = 0.0
            aucSumList[trainMode] = 0.0

        print 'running 5-fold CV...'
        if labelMode == 1:
            title = 'totalGroup10'
        else:
            title = 'totalGroup2'
        for roundNum in range(5):
            indexFile = open('dataset/experiment/groups/' + title + '/indices/' + str(roundNum)+ '.indices', 'r')
            for line in indexFile:
                temp = json.loads(line.strip())
                trainIDs = temp['train']
                testIDs = temp['test']
            indexFile.close()
            label_test = []
            label_train = []
            if trainIDs[0] in errorIDList:
                print 'first trainID is invalid!'
            else:
                feature_train = featureData[trainIDs[0]]
            if testIDs[0] in errorIDList:
                print 'first testID is invalid!'
            else:
                feature_test = featureData[testIDs[0]]
            for j, tweetID in enumerate(trainIDs):
                if tweetID not in errorIDList:
                    if j > 0:
                        feature_train = vstack((feature_train, featureData[tweetID]), format='csr')
                    label_train.append(labelData[tweetID])
            for j, tweetID in enumerate(testIDs):
                if tweetID not in errorIDList:
                    if j > 0:
                        feature_test = vstack((feature_test, featureData[tweetID]), format='csr')
                    label_test.append(labelData[tweetID])

            print 'Round: '+str(roundNum)
            for trainMode in trainModeList:
                print trainMode
                if trainMode == 'MaxEnt':
                    model = linear_model.LogisticRegression()
                elif trainMode == 'NaiveBayes':
                    model = MultinomialNB()
                elif trainMode == 'SVM':
                    model = svm.SVC(probability=True)
                else:
                    model = MLPClassifier(activation='logistic', learning_rate='constant')

                print 'Training...'
                model.fit(feature_train, label_train)
                print 'Inference...'
                predictions = model.predict(feature_test)

                if recordProb:
                    if labelMode == 1:
                        trainOutputFile = open('outputs/totalGroup10_'+str(splitNum)+'/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'w')
                        testOutputFile = open('outputs/totalGroup10_'+str(splitNum)+'/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'w')
                    else:
                        trainOutputFile = open('outputs/totalGroup2_'+str(splitNum)+'/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'w')
                        testOutputFile = open('outputs/totalGroup2_'+str(splitNum)+'/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'w')
                    trainProb = model.predict_proba(feature_train)
                    testProb = model.predict_proba(feature_test)
                    labelList = model.classes_
                    testOutput = {}
                    for num, probs in enumerate(testProb):
                        temp = {}
                        for index, prob in enumerate(probs):
                            inferTopic = labelList[index]
                            temp[inferTopic] = prob
                        sorted_temp = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
                        outList = {}
                        for topic, prob in sorted_temp:
                            outList[topic] = prob
                        testOutput[testIDs[num]] = outList
                    testOutputFile.write(json.dumps(testOutput))
                    trainOutput = {}
                    for num, probs in enumerate(trainProb):
                        temp = {}
                        for index, prob in enumerate(probs):
                            inferTopic = labelList[index]
                            temp[inferTopic] = prob
                        sorted_temp = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
                        outList = {}
                        for topic, prob in sorted_temp:
                            outList[topic] = prob
                        trainOutput[trainIDs[num]] = outList
                    trainOutputFile.write(json.dumps(trainOutput))
                    testOutputFile.close()
                    trainOutputFile.close()

                precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

                precisionSumList[trainMode] += precision
                recallSumList[trainMode] += recall
                F1SumList[trainMode] += F1
                aucSumList[trainMode] += auc

        outputF1List = {}
        outputPrecisionList = {}
        outputRecallList = {}
        outputAUCList = {}
        for trainMode in trainModeList:
            print trainMode
            outputF1List[trainMode] = F1SumList[trainMode] / 5
            outputPrecisionList[trainMode] = precisionSumList[trainMode] / 5
            outputRecallList[trainMode] = recallSumList[trainMode] / 5
            outputAUCList[trainMode] = aucSumList[trainMode] / 5

            print outputPrecisionList[trainMode]
            print outputRecallList[trainMode]
            print outputF1List[trainMode]
            print outputAUCList[trainMode]

        print ''
        for trainMode in trainModeList:
            resultFile.write(trainMode)
            resultFile.write(str(outputPrecisionList[trainMode]) + '\n')
            resultFile.write(str(outputRecallList[trainMode]) + '\n')
            resultFile.write(str(outputF1List[trainMode]) + '\n')
            resultFile.write(str(outputAUCList[trainMode]) + '\n')
            resultFile.write('\n')
            resultFile.flush()

    resultFile.close()


# vectorMode 1: tfidf, 2: binaryCount
# featureMode 0: semantic only, 1: vector only, 2: both
def runModel2(groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, ablationIndex, removeOutliers=True):
    outputFile = 'results/ablation/' + groupTitle
    resultFile = open(outputFile, 'a')

    print groupTitle
    resultFile.write(groupTitle + '_' + str(groupSize)+ '\n')
    resultFile.write('Label Mode: ' + str(labelMode) + '\n')

    print 'loading data...'
    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    labels = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    if removeOutliers:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    else:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '_full.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        ids.append(str(item['id']))
        contents.append(item['content'])
        if labelMode == 1:
            labels.append(item['label'])
        elif labelMode == 2:
            if item['label'] > splitNum:
                labels.append(1)
            else:
                labels.append(0)
        if contentFeatures:
            time.append(utilities.hourMapper(item['hour']))
            usernames.append(item['mentions'])
            days.append(dayMapper[item['day']])
            authorFollowers.append(int(item['author_followers_count']))
            authorStatusCount.append(int(item['author_statuses_count']))
            authorFavoriteCount.append(int(item['author_favorite_count']))
            authorListedCount.append(int(item['author_listed_count']))
            authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        resultFile.write('tfidf \n')
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        resultFile.write('binary count \n')
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)
    else:
        resultFile.write('no vector features \n')

    if embeddingMode == 1:
        print 'generating CMU embeddings...'
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
        resultFile.write('CMU Twitter Hashtag embedding'+'\n')
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
        resultFile.write('MIT Twitter Embedding'+'\n')
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)
        resultFile.write('gneral twitter word2vec + CBOW' + '\n')
    else:
        resultFile.write('no twitter embedding'+'\n')

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(labels):
            print 'Embedding vector size error...'

    if contentFeatures:
        resultFile.write('with content features'+'\n')
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index]/authorIntervals[index])
                temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                temp.append(authorListedCount[index]/authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))
    else:
        resultFile.write('no content features'+'\n')

    if featureMode == 0:
        resultFile.write('content features only \n')
        features = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        resultFile.write('vector features only \n')
        features = vectorMatrix
    elif featureMode == 2:
        resultFile.write('content and embedding'+'\n')
        features = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        resultFile.write('embedding only'+'\n')
        features = csr_matrix(embeddingFeature)
    else:
        resultFile.write('vector and content features \n')
        features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    resultFile.flush()

    featureData = {}
    labelData = {}
    for lineNum, label in enumerate(labels):
        labelData[ids[lineNum]] = label
        featureData[ids[lineNum]] = features.getrow(lineNum)

    precisionSumList = {}
    recallSumList = {}
    F1SumList = {}
    aucSumList = {}
    for trainMode in trainModeList:
        precisionSumList[trainMode] = 0.0
        recallSumList[trainMode] = 0.0
        F1SumList[trainMode] = 0.0
        aucSumList[trainMode] = 0.0

    print 'running 5-fold CV...'
    for roundNum in range(5):
        if removeOutliers:
            indexFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '.'+str(roundNum), 'r')
        else:
            indexFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '_full.'+str(roundNum), 'r')
        for line in indexFile:
            temp = json.loads(line.strip())
            trainIDs = temp['train']
            testIDs = temp['test']
        indexFile.close()
        label_test = []
        label_train = []
        if trainIDs[0] in errorIDList:
            print 'first trainID is invalid!'
        else:
            feature_train = featureData[trainIDs[0]]
        if testIDs[0] in errorIDList:
            print 'first testID is invalid!'
        else:
            feature_test = featureData[testIDs[0]]
        for j, tweetID in enumerate(trainIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_train = vstack((feature_train, featureData[tweetID]), format='csr')
                label_train.append(labelData[tweetID])
        for j, tweetID in enumerate(testIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_test = vstack((feature_test, featureData[tweetID]), format='csr')
                label_test.append(labelData[tweetID])

        print 'Round: '+str(roundNum)
        for trainMode in trainModeList:
            print trainMode
            if trainMode == 'MaxEnt':
                model = linear_model.LogisticRegression()
            elif trainMode == 'NaiveBayes':
                model = MultinomialNB()
            elif trainMode == 'SVM':
                model = svm.SVC(probability=True)
            else:
                model = MLPClassifier(activation='logistic', learning_rate='constant')

            print 'Training...'
            model.fit(feature_train, label_train)
            print 'Inference...'
            predictions = model.predict(feature_test)

            precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

            precisionSumList[trainMode] += precision
            recallSumList[trainMode] += recall
            F1SumList[trainMode] += F1
            aucSumList[trainMode] += auc

    outputF1List = {}
    outputPrecisionList = {}
    outputRecallList = {}
    outputAUCList = {}
    for trainMode in trainModeList:
        print trainMode
        outputF1List[trainMode] = F1SumList[trainMode] / 5
        outputPrecisionList[trainMode] = precisionSumList[trainMode] / 5
        outputRecallList[trainMode] = recallSumList[trainMode] / 5
        outputAUCList[trainMode] = aucSumList[trainMode] / 5

        print outputPrecisionList[trainMode]
        print outputRecallList[trainMode]
        print outputF1List[trainMode]
        print outputAUCList[trainMode]

    print ''
    for trainMode in trainModeList:
        resultFile.write(trainMode+'\n')
        resultFile.write(str(outputPrecisionList[trainMode]) + '\n')
        resultFile.write(str(outputRecallList[trainMode]) + '\n')
        resultFile.write(str(outputF1List[trainMode]) + '\n')
        resultFile.write(str(outputAUCList[trainMode]) + '\n')
        resultFile.write('\n')
        resultFile.flush()

    resultFile.close()


def runModelKeyword(keyword, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, ablationIndex):
    print keyword
    print 'loading data...'
    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    labels = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    inputFile = open('dataset/experiment/' + keyword + '.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        ids.append(str(item['id']))
        contents.append(item['content'])
        if labelMode == 1:
            labels.append(item['label'])
        elif labelMode == 2:
            if item['label'] > splitNum:
                labels.append(1)
            else:
                labels.append(0)
        if contentFeatures:
            time.append(utilities.hourMapper(item['hour']))
            usernames.append(item['mentions'])
            days.append(dayMapper[item['day']])
            authorFollowers.append(int(item['author_followers_count']))
            authorStatusCount.append(int(item['author_statuses_count']))
            authorFavoriteCount.append(int(item['author_favorite_count']))
            authorListedCount.append(int(item['author_listed_count']))
            authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    if embeddingMode == 1:
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(labels):
            print 'Embedding vector size error...'

    if contentFeatures:
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index]/authorIntervals[index])
                temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                temp.append(authorListedCount[index]/authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))

    if featureMode == 0:
        features = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        features = vectorMatrix
    elif featureMode == 2:
        features = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        features = csr_matrix(embeddingFeature)
    else:
        features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')

    featureData = {}
    labelData = {}
    for lineNum, label in enumerate(labels):
        labelData[ids[lineNum]] = label
        featureData[ids[lineNum]] = features.getrow(lineNum)

    precisionSumList = {}
    recallSumList = {}
    F1SumList = {}
    aucSumList = {}
    for trainMode in trainModeList:
        precisionSumList[trainMode] = 0.0
        recallSumList[trainMode] = 0.0
        F1SumList[trainMode] = 0.0
        aucSumList[trainMode] = 0.0

    print 'running 5-fold CV...'
    kf = KFold(n_splits=5, shuffle=True)
    for roundNum, (train_indices, test_indices) in enumerate(kf.split(ids)):
        trainIDs = []
        testIDs = []
        for index in train_indices:
            trainIDs.append(ids[index])
        for index in test_indices:
            testIDs.append(ids[index])

        label_test = []
        label_train = []
        if trainIDs[0] in errorIDList:
            print 'first trainID is invalid!'
        else:
            feature_train = featureData[trainIDs[0]]
        if testIDs[0] in errorIDList:
            print 'first testID is invalid!'
        else:
            feature_test = featureData[testIDs[0]]
        for j, tweetID in enumerate(trainIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_train = vstack((feature_train, featureData[tweetID]), format='csr')
                label_train.append(labelData[tweetID])
        for j, tweetID in enumerate(testIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_test = vstack((feature_test, featureData[tweetID]), format='csr')
                label_test.append(labelData[tweetID])

        print 'Round: '+str(roundNum)
        for trainMode in trainModeList:
            print trainMode
            if trainMode == 'MaxEnt':
                model = linear_model.LogisticRegression()
            elif trainMode == 'NaiveBayes':
                model = MultinomialNB()
            elif trainMode == 'SVM':
                model = svm.SVC(probability=True)
            else:
                model = MLPClassifier(activation='logistic', learning_rate='constant')

            print 'Training...'
            model.fit(feature_train, label_train)
            print 'Inference...'
            predictions = model.predict(feature_test)

            precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

            precisionSumList[trainMode] += precision
            recallSumList[trainMode] += recall
            F1SumList[trainMode] += F1
            aucSumList[trainMode] += auc

    outputF1List = {}
    outputPrecisionList = {}
    outputRecallList = {}
    outputAUCList = {}
    for trainMode in trainModeList:
        print trainMode
        outputF1List[trainMode] = F1SumList[trainMode] / 5
        outputPrecisionList[trainMode] = precisionSumList[trainMode] / 5
        outputRecallList[trainMode] = recallSumList[trainMode] / 5
        outputAUCList[trainMode] = aucSumList[trainMode] / 5

        print outputPrecisionList[trainMode]
        print outputRecallList[trainMode]
        print outputF1List[trainMode]
        print outputAUCList[trainMode]
    print ''


def runModelGroup(groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, ablationIndex, groupList):
    if -1 in groupList:
        groupList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print 'loading data...'
    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    labels = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    inputFile = open('dataset/experiment/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        if item['group'] in groupList:
            ids.append(str(item['id']))
            contents.append(item['content'])
            if labelMode == 1:
                labels.append(item['label'])
            elif labelMode == 2:
                if item['label'] > splitNum:
                    labels.append(1)
                else:
                    labels.append(0)
            if contentFeatures:
                time.append(utilities.hourMapper(item['hour']))
                usernames.append(item['mentions'])
                days.append(dayMapper[item['day']])
                authorFollowers.append(int(item['author_followers_count']))
                authorStatusCount.append(int(item['author_statuses_count']))
                authorFavoriteCount.append(int(item['author_favorite_count']))
                authorListedCount.append(int(item['author_listed_count']))
                authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    if embeddingMode == 1:
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(labels):
            print 'Embedding vector size error...'

    if contentFeatures:
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index]/authorIntervals[index])
                temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                temp.append(authorListedCount[index]/authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))

    if featureMode == 0:
        features = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        features = vectorMatrix
    elif featureMode == 2:
        features = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        features = csr_matrix(embeddingFeature)
    else:
        features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')

    featureData = {}
    labelData = {}
    for lineNum, label in enumerate(labels):
        labelData[ids[lineNum]] = label
        featureData[ids[lineNum]] = features.getrow(lineNum)

    precisionSumList = {}
    recallSumList = {}
    F1SumList = {}
    aucSumList = {}
    for trainMode in trainModeList:
        precisionSumList[trainMode] = 0.0
        recallSumList[trainMode] = 0.0
        F1SumList[trainMode] = 0.0
        aucSumList[trainMode] = 0.0

    print 'running 5-fold CV...'
    kf = KFold(n_splits=5, shuffle=True)
    for roundNum, (train_indices, test_indices) in enumerate(kf.split(ids)):
        trainIDs = []
        testIDs = []
        for index in train_indices:
            trainIDs.append(ids[index])
        for index in test_indices:
            testIDs.append(ids[index])

        label_test = []
        label_train = []
        if trainIDs[0] in errorIDList:
            print 'first trainID is invalid!'
        else:
            feature_train = featureData[trainIDs[0]]
        if testIDs[0] in errorIDList:
            print 'first testID is invalid!'
        else:
            feature_test = featureData[testIDs[0]]
        for j, tweetID in enumerate(trainIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_train = vstack((feature_train, featureData[tweetID]), format='csr')
                label_train.append(labelData[tweetID])
        for j, tweetID in enumerate(testIDs):
            if tweetID not in errorIDList:
                if j > 0:
                    feature_test = vstack((feature_test, featureData[tweetID]), format='csr')
                label_test.append(labelData[tweetID])

        print 'Round: '+str(roundNum)
        for trainMode in trainModeList:
            print trainMode
            if trainMode == 'MaxEnt':
                model = linear_model.LogisticRegression()
            elif trainMode == 'NaiveBayes':
                model = MultinomialNB()
            elif trainMode == 'SVM':
                model = svm.SVC(probability=True)
            else:
                model = MLPClassifier(activation='logistic', learning_rate='constant')

            print 'Training...'
            model.fit(feature_train, label_train)
            print 'Inference...'
            predictions = model.predict(feature_test)

            precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

            precisionSumList[trainMode] += precision
            recallSumList[trainMode] += recall
            F1SumList[trainMode] += F1
            aucSumList[trainMode] += auc

    outputF1List = {}
    outputPrecisionList = {}
    outputRecallList = {}
    outputAUCList = {}
    for trainMode in trainModeList:
        print trainMode
        outputF1List[trainMode] = F1SumList[trainMode] / 5
        outputPrecisionList[trainMode] = precisionSumList[trainMode] / 5
        outputRecallList[trainMode] = recallSumList[trainMode] / 5
        outputAUCList[trainMode] = aucSumList[trainMode] / 5

        print outputPrecisionList[trainMode]
        print outputRecallList[trainMode]
        print outputF1List[trainMode]
        print outputAUCList[trainMode]
    print ''


def runModelGroup2(groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, ablationIndex, trainGroupList, testGroupList):
    print 'loading data...'
    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    label_train = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    inputFile = open('dataset/experiment/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        if item['group'] in trainGroupList:
            ids.append(str(item['id']))
            contents.append(item['content'])
            if labelMode == 1:
                label_train.append(item['label'])
            elif labelMode == 2:
                if item['label'] > splitNum:
                    label_train.append(1)
                else:
                    label_train.append(0)
            if contentFeatures:
                time.append(utilities.hourMapper(item['hour']))
                usernames.append(item['mentions'])
                days.append(dayMapper[item['day']])
                authorFollowers.append(int(item['author_followers_count']))
                authorStatusCount.append(int(item['author_statuses_count']))
                authorFavoriteCount.append(int(item['author_favorite_count']))
                authorListedCount.append(int(item['author_listed_count']))
                authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    if embeddingMode == 1:
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(label_train):
            print 'Embedding vector size error...'

    if contentFeatures:
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index]/authorIntervals[index])
                temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                temp.append(authorListedCount[index]/authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))

    if featureMode == 0:
        feature_train = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        feature_train = vectorMatrix
    elif featureMode == 2:
        feature_train = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        feature_train = csr_matrix(embeddingFeature)
    else:
        feature_train = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')

    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    label_test = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    inputFile = open('dataset/experiment/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        if item['group'] in testGroupList:
            ids.append(str(item['id']))
            contents.append(item['content'])
            if labelMode == 1:
                label_test.append(item['label'])
            elif labelMode == 2:
                if item['label'] > splitNum:
                    label_test.append(1)
                else:
                    label_test.append(0)
            if contentFeatures:
                time.append(utilities.hourMapper(item['hour']))
                usernames.append(item['mentions'])
                days.append(dayMapper[item['day']])
                authorFollowers.append(int(item['author_followers_count']))
                authorStatusCount.append(int(item['author_statuses_count']))
                authorFavoriteCount.append(int(item['author_favorite_count']))
                authorListedCount.append(int(item['author_listed_count']))
                authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    if embeddingMode == 1:
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(label_train):
            print 'Embedding vector size error...'

    if contentFeatures:
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index] / authorIntervals[index])
                temp.append(authorFavoriteCount[index] / authorStatusCount[index])
                temp.append(authorListedCount[index] / authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))

    if featureMode == 0:
        feature_test = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        feature_test = vectorMatrix
    elif featureMode == 2:
        feature_test = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        feature_test = csr_matrix(embeddingFeature)
    else:
        feature_test = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')


    precisionSumList = {}
    recallSumList = {}
    F1SumList = {}
    aucSumList = {}
    for trainMode in trainModeList:
        precisionSumList[trainMode] = 0.0
        recallSumList[trainMode] = 0.0
        F1SumList[trainMode] = 0.0
        aucSumList[trainMode] = 0.0

    for trainMode in trainModeList:
        print trainMode
        if trainMode == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainMode == 'NaiveBayes':
            model = MultinomialNB()
        elif trainMode == 'SVM':
            model = svm.SVC(probability=True)
        else:
            model = MLPClassifier(activation='logistic', learning_rate='constant')

        print 'Training...'
        model.fit(feature_train, label_train)
        print 'Inference...'
        predictions = model.predict(feature_test)

        precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

        precisionSumList[trainMode] += precision
        recallSumList[trainMode] += recall
        F1SumList[trainMode] += F1
        aucSumList[trainMode] += auc

    outputF1List = {}
    outputPrecisionList = {}
    outputRecallList = {}
    outputAUCList = {}
    for trainMode in trainModeList:
        print trainMode
        outputF1List[trainMode] = F1SumList[trainMode] / 5
        outputPrecisionList[trainMode] = precisionSumList[trainMode] / 5
        outputRecallList[trainMode] = recallSumList[trainMode] / 5
        outputAUCList[trainMode] = aucSumList[trainMode] / 5

        print outputPrecisionList[trainMode]
        print outputRecallList[trainMode]
        print outputF1List[trainMode]
        print outputAUCList[trainMode]
    print ''


def validateFull(trainMode, labelMode, featureMode, splitNum):
    inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
    ids = []
    labels = []
    for line in inputFile:
        item = json.loads(line.strip())
        ids.append(str(item['id']))
        if labelMode == 1:
            labels.append(item['label'])
        elif labelMode == 2:
            if item['label'] > splitNum:
                labels.append(1)
            else:
                labels.append(0)
    inputFile.close()

    labelData = {}
    for lineNum, label in enumerate(labels):
        labelData[ids[lineNum]] = label

    if labelMode == 1:
        distFile = open('outputs/totalGroup10_' + str(splitNum) + '_full/' + trainMode + '/' + str(featureMode) + '.prob', 'r')
    else:
        distFile = open('outputs/totalGroup2_' + str(splitNum) + '_full/' + trainMode + '/' + str(featureMode) + '.prob', 'r')

    for line in distFile:
        data = json.loads(line.strip())

    predictions = []
    label_test = []
    for id, dist in data.items():
        label_test.append(int(labelData[id]))
        predictions.append(int(max(dist.iteritems(), key=operator.itemgetter(1))[0]))

    precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)
    print str(trainMode)+'\t' +str(labelMode)+'\t' +str(featureMode)+'\t' +str(splitNum)
    print precision
    print recall
    print F1
    print auc
    print ''


def runModelDiff(groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, ablationIndex, removeOutliers=True):
    print groupTitle
    print 'loading data...'
    errorIDList = []
    ids = []
    contents = []
    days = []
    time = []
    labels = []
    usernames = []
    additionalFeatures = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []

    if removeOutliers:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '.labeled', 'r')
    else:
        inputFile = open('dataset/experiment/labeled_data/' + groupTitle + '_' + str(groupSize) + '_full.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        ids.append(str(item['id']))
        contents.append(item['content'])
        if labelMode == 1:
            labels.append(item['label'])
        elif labelMode == 2:
            if item['label'] > splitNum:
                labels.append(1)
            else:
                labels.append(0)
        if contentFeatures:
            time.append(utilities.hourMapper(item['hour']))
            usernames.append(item['mentions'])
            days.append(dayMapper[item['day']])
            authorFollowers.append(int(item['author_followers_count']))
            authorStatusCount.append(int(item['author_statuses_count']))
            authorFavoriteCount.append(int(item['author_favorite_count']))
            authorListedCount.append(int(item['author_listed_count']))
            authorIntervals.append(int(item['authorInterval']))
    inputFile.close()

    if vectorMode == 1:
        print 'Generating tfidf vectors...'
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
        vectorMatrix = vectorizer.fit_transform(contents)
    elif vectorMode == 2:
        print 'Generating binary ngram vectors...'
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                     binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    if embeddingMode == 1:
        print 'generating CMU embeddings...'
        totalFile = open('dataset/experiment/total.json', 'r')
        totalIDList = []
        for line in totalFile:
            temp = json.loads(line.strip())
            totalIDList.append(str(temp['id']))
        totalFile.close()
        embeddings = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        embFeatures = {}
        for index, tweetID in enumerate(totalIDList):
            embFeatures[tweetID] = embeddings[index]
        tempFeatures = []
        for tweetID in ids:
            tempFeatures.append(embFeatures[tweetID])
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = simpleTokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = utilities.content2vec(embModel, content)
            if tweetVec == None:
                errorIDList.append(id)
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    if embeddingMode in [1, 2, 3]:
        if len(embeddingFeature) != len(labels):
            print 'Embedding vector size error...'

    if contentFeatures:
        mentionMapper = utilities.mapMention('dataset/experiment/mention.json')
        afinn = Afinn()
        parseLength = {}
        headCount = {}
        POScounts = {}
        lengthFile = open('dataset/experiment/parser/total.length', 'r')
        headCountFile = open('dataset/experiment/parser/total.headCount', 'r')
        posCountFile = open('dataset/experiment/parser/total.posCount', 'r')
        for line in lengthFile:
            temp = line.strip().split(' :: ')
            parseLength[temp[1]] = int(temp[0])
        for line in headCountFile:
            temp = line.strip().split(' :: ')
            headCount[temp[1]] = int(temp[0])
        for line in posCountFile:
            temp = line.strip().split(' :: ')
            POScounts[temp[1]] = utilities.POSRatio(temp[0].split(' '))
        headCountFile.close()
        lengthFile.close()
        posCountFile.close()

        for index, content in enumerate(contents):
            tweetID = ids[index]
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            readScore = textstat.coleman_liau_index(content)

            if ablationIndex != 0:
                temp.append(sentiScore / twLen)
            if ablationIndex != 1:
                temp.append(twLen)
                temp.append(readScore)
                temp.append(parseLength[tweetID] / twLen)
                temp.append(headCount[tweetID] / twLen)
            if ablationIndex != 2:
                temp.append(authorStatusCount[index]/authorIntervals[index])
                temp.append(authorFavoriteCount[index]/authorStatusCount[index])
                temp.append(authorListedCount[index]/authorFollowers[index])
            if ablationIndex != 3:
                temp.append(days[index])
                temp.append(time[index])
            if ablationIndex != 4:
                if any(char.isdigit() for char in content):
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 5:
                temp += POScounts[tweetID]
            if ablationIndex != 6:
                # temp.append(content.count('URRL'))
                if content.count('http://URL') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('HHTTG'))
                if content.count('#HTG') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('USSERNM'))
                if content.count('@URNM') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 7:
                # temp.append(content.count('!'))
                if content.count('!') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
                # temp.append(content.count('?'))
                if content.count('?') > 0:
                    temp.append(1)
                else:
                    temp.append(0)
            if ablationIndex != 8:
                mentionFlag = 0
                mentionFollowers = 0
                userCount = 0.0
                for user in usernames[index]:
                    if user in mentionMapper:
                        userCount += 1
                        if mentionMapper[user][0] == 1:
                            mentionFlag = 1
                        mentionFollowers += mentionMapper[user][1]
                temp.append(mentionFlag)

                if userCount == 0:
                    temp.append(0.0)
                else:
                    temp.append(mentionFollowers / userCount)

            additionalFeatures.append(numpy.array(temp))

    if featureMode == 0:
        features = csr_matrix(numpy.array(additionalFeatures))
    elif featureMode == 1:
        features = vectorMatrix
    elif featureMode == 2:
        features = hstack((embeddingFeature, csr_matrix(numpy.array(additionalFeatures))), format='csr')
    elif featureMode == 3:
        features = csr_matrix(embeddingFeature)
    else:
        features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')

    featureData = {}
    labelData = {}
    contentData = {}
    for lineNum, label in enumerate(labels):
        labelData[ids[lineNum]] = label
        featureData[ids[lineNum]] = features.getrow(lineNum)
        contentData[ids[lineNum]] = contents[lineNum]

    print 'running prediction...'
    roundNum = 1
    if removeOutliers:
        indexFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '.'+str(roundNum), 'r')
    else:
        indexFile = open('dataset/experiment/fold_indicies/' + groupTitle + '_' + str(groupSize) + '_full.'+str(roundNum), 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()
    label_test = []
    label_train = []
    content_test = []
    if trainIDs[0] in errorIDList:
        print 'first trainID is invalid!'
    else:
        feature_train = featureData[trainIDs[0]]
    if testIDs[0] in errorIDList:
        print 'first testID is invalid!'
    else:
        feature_test = featureData[testIDs[0]]
    for j, tweetID in enumerate(trainIDs):
        if tweetID not in errorIDList:
            if j > 0:
                feature_train = vstack((feature_train, featureData[tweetID]), format='csr')
            label_train.append(labelData[tweetID])
    for j, tweetID in enumerate(testIDs):
        if tweetID not in errorIDList:
            if j > 0:
                feature_test = vstack((feature_test, featureData[tweetID]), format='csr')
            label_test.append(labelData[tweetID])
            content_test.append(contentData[tweetID]+'\t'+str(tweetID))

    print 'Round: '+str(roundNum)
    for trainMode in trainModeList:
        print trainMode
        if trainMode == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainMode == 'NaiveBayes':
            model = MultinomialNB()
        elif trainMode == 'SVM':
            model = svm.SVC(probability=True)
        else:
            model = MLPClassifier(activation='logistic', learning_rate='constant')

        print 'Training...'
        model.fit(feature_train, label_train)
        print 'Inference...'
        predictions = model.predict(feature_test)

        #precision, recall, F1, auc = utilities.evaluate(predictions, label_test, labelMode, splitNum=splitNum)

        outContentFile = open('outputs/analysis/content.'+str(featureMode), 'w')
        outTrueLabelFile = open('outputs/analysis/trueLabel.'+str(featureMode), 'w')
        outPredictLabelFile = open('outputs/analysis/predLabel.'+str(featureMode), 'w')

        for content in content_test:
            outContentFile.write(content+'\n')
        for label in label_test:
            outTrueLabelFile.write(str(label)+'\n')
        for label in predictions:
            outPredictLabelFile.write(str(label)+'\n')

        outContentFile.close()
        outTrueLabelFile.close()
        outPredictLabelFile.close()


if __name__ == "__main__":
    #groupSize, groupTitle, vectorMode, featureMode, contentFeatures, embeddingMode, trainModeList, labelMode, splitNum, recordProb, ablationIndex
    # [vectorMode] 1: tfidf, 2: binaryCount, 4: None
    # [featureMode] 0: content only, 1: ngram only, 2: content and embedding, 3: embedding only, 4: content and ngram
    # [embeddingMode] 1: CMU Hashtag embedding, 2: doc2vec, 3: word2vec+CBOW
    # [ablation analysis] 100: all features
    #writeLabels(1, 'totalGroup', 1)

    #shuffleTweets2(2.4, 'topicGroup', True)

    #trainModeList = ['MaxEnt']
    #runModel2(2.4, 'simGroup_binary', 2, 1, False, 0, trainModeList, 2, 5, 100, True)


    trainModeList = ['SVM']
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 0, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 1, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 2, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 3, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 4, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 5, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 6, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 7, True)
    #runModel2(5.4, 'simGroup_emb', 4, 0, True, 0, trainModeList, 2, 5, 8, True)

    #runModelGroup2(5, 'simGroup_binary', 4, 0, True, 0, trainModeList, 2, 5, 100, [0, 1, 2, 4], [3])

    runModelDiff(5.4, 'simGroup_emb', 2, 1, False, 0, ['MaxEnt'], 2, 5, 100, True)
    runModelDiff(5.4, 'simGroup_emb', 4, 0, True, 0, ['SVM'], 2, 5, 100, True)
    runModelDiff(5.4, 'simGroup_emb', 4, 3, False, 1, ['SVM'], 2, 5, 100, True)

    #runModelKeyword('iphone', 2, 1, False, 0, trainModeList, 2, 5, 100)

    #shuffleTweetsSimple(True)