__author__ = 'rencui'
import json
from afinn import Afinn
import subprocess
import numpy
import utilities
from textstat.textstat import textstat
from sklearn.feature_extraction.text import *
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from tokenizer import simpleTokenize
from scipy.sparse import hstack, csr_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import operator
from sklearn.model_selection import KFold

reload(sys)
sys.setdefaultencoding('utf8')

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}

stemmer = PorterStemmer()


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


# vectorMode 1: tfidf, 2: binaryCount
# featureMode 0: semantic only, 1: vector only, 2: both
def runModel(groupSize, groupTitle, vectorMode, featureMode, trainMode, labelMode, recordProb, ablationIndex):
    outputFile = 'results/' + groupTitle + '_' + trainMode + '.'+str(vectorMode)+'_'+str(featureMode)
    resultFile = open(outputFile, 'a')
    mentionMapper = utilities.mapMention('dataset/experiment/mention.json')

    print groupTitle
    print trainMode
    resultFile.write(groupTitle + '\n')
    for group in range(groupSize):
        print 'group: ' + str(group)
        resultFile.write('group: ' + str(group) + '\n')
        # happy_log_probs, sad_log_probs = utilities.readSentimentList('twitter_sentiment_list.csv')
        afinn = Afinn()
        inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.labeled', 'r')
        lengthFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.length', 'r')
        headCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.headCount', 'r')
        posCountFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.posCount', 'r')

        ids = []
        contents = []
        scores = []
        days = []
        time = []
        labels = []
        parseLength = []
        headCount = []
        usernames = []
        additionalFeatures = []
        classes = []
        POScounts = []
        authorFollowers = []
        authorStatusCount = []
        authorFavoriteCount = []
        authorListedCount = []
        authorIntervals = []

        print 'loading...'

        for line in inputFile:
            item = json.loads(line.strip())
            time.append(utilities.hourMapper(item['hour']))
            ids.append(item['id'])
            usernames.append(item['mentions'])
            days.append(dayMapper[item['day']])
            contents.append(item['content'])
            scores.append(float(item['score']))
            authorFollowers.append(int(item['author_followers_count']))
            authorStatusCount.append(int(item['author_statuses_count']))
            authorFavoriteCount.append(int(item['author_favorite_count']))
            authorListedCount.append(int(item['author_listed_count']))
            authorIntervals.append(int(item['authorInterval']))
            if labelMode == 1:
                labels.append(item['label'])
            else:
                if item['label'] > 7:
                    labels.append(1)
                else:
                    labels.append(0)

        distMapper = {}

        if vectorMode == 1:
            resultFile.write('tfidf \n')
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english')
            vectorMatrix = vectorizer.fit_transform(contents)
        elif vectorMode == 2:
            resultFile.write('binary count \n')
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                         binary='True')
            vectorMatrix = vectorizer.fit_transform(contents)
        elif vectorMode == 3:
            listFile = open('LDA/LDAinput.list', 'r')
            idMapper = {}
            for index, line in enumerate(listFile):
                idMapper[index] = line.strip()
            subprocess.check_output('java -Xmx1024m -jar LDA/tmt-0.4.0.jar LDA/assign2.scala', shell=True)
            distFile = open('LDA/TMTSnapshots/document-topic-distributions.csv', 'r')
            for line in distFile:
                seg = line.strip().split(',')
                outList = []
                for item in seg[1:]:
                    outList.append(float(item))
                distMapper[idMapper[int(seg[0])]] = outList
            distFile.close()
        else:
            resultFile.write('no vector features \n')

        for line in lengthFile:
            parseLength.append(int(line.strip(' :: ')[0]))
        for line in headCountFile:
            headCount.append(int(line.strip(' :: ')[0]))
        for line in posCountFile:
            POScounts.append(utilities.POSRatio(line.strip().split(' :: ')[0].split(' ')))

        headCountFile.close()
        lengthFile.close()
        posCountFile.close()
        inputFile.close()

        for index, content in enumerate(contents):
            temp = []
            words = simpleTokenize(content)
            twLen = float(len(words))
            sentiScore = afinn.score(content)
            # posProb, negProb = utilities.classifySentiment(words, happy_log_probs, sad_log_probs)
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
            classes.append(labels[index])

        if featureMode == 0:
            resultFile.write('content features only \n')
            features = csr_matrix(numpy.array(additionalFeatures))
        elif featureMode == 1:
            resultFile.write('vector features only \n')
            features = vectorMatrix
        elif featureMode == 2:
            resultFile.write('embedding features only \n')
            embedFeatures = []
            for id in ids:
                embedFeatures.append(numpy.array(distMapper[id]))
            features = csr_matrix(numpy.array(embedFeatures))
        elif featureMode == 3:
            resultFile.write('embedding and semantic features only \n')
            embedFeatures = []
            for id in ids:
                embedFeatures.append(numpy.array(distMapper[id]))
            features = hstack((csr_matrix(numpy.array(additionalFeatures)), csr_matrix(numpy.array(embedFeatures))), format='csr')
        else:
            resultFile.write('vector and semantic features \n')
            features = hstack((vectorMatrix, csr_matrix(numpy.array(additionalFeatures))), format='csr')

        precisionSum = 0.0
        recallSum = 0.0
        F1Sum = 0.0
        R2Sum = 0.0
        aucSum = 0.0
        MSESum = 0.0
        accuracy1Sum = 0.0
        accuracy2Sum = 0.0
        resultFile.flush()

        print 'running 5-fold CV...'
        roundNum = 0
        kf = KFold(5)
        for train_indices, test_indices in kf.split(classes):
            feature_train, feature_test = features[train_indices], features[test_indices]
            label_train = []
            label_test = []
            for train_index in train_indices:
                label_train.append(classes[train_index])
            for test_index in test_indices:
                label_test.append(classes[test_index])

            print 'Round: '+str(roundNum)
            roundNum += 1
            #feature_train, feature_test, label_train, label_test = train_test_split(features, classes, test_size=0.2, random_state=0)
            #feature_train, feature_test, label_train, label_test = dataSplit(features, classes, i)
            if trainMode == 'MaxEnt':
                # model = MLPClassifier(algorithm='sgd', activation='logistic', learning_rate_init=0.02, learning_rate='constant', batch_size=10)
                model = linear_model.LogisticRegression()
            elif trainMode == 'NaiveBayes':
                model = MultinomialNB()
            elif trainMode == 'RF':
                model = ExtraTreesClassifier(n_estimators=50, random_state=0)
            elif trainMode == 'Ada':
                model = AdaBoostClassifier()
            elif trainMode == 'LR':
                model = linear_model.LinearRegression()
            elif trainMode == 'SVR':
                model = svm.SVR()
            elif trainMode == 'Pass':
                model = linear_model.PassiveAggressiveRegressor()
            elif trainMode == 'SVM':
                model = svm.SVC(probability=True)
            else:
                model = MLPClassifier(activation='logistic', learning_rate_init=0.02, learning_rate='constant', batch_size=50)

            model.fit(feature_train, label_train)
            predictions = model.predict(feature_test)

            if recordProb:
                trainOutputFile = open('outputs/' + groupTitle + '/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'w')
                testOutputFile = open('outputs/' + groupTitle + '/' + trainMode + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'w')
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
                    testOutput[num] = outList
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
                    trainOutput[num] = outList
                trainOutputFile.write(json.dumps(trainOutput))
                testOutputFile.close()
                trainOutputFile.close()

            # binaryPreds = scoreToBinary(predictions)
            # binaryTestLabels = scoreToBinary(label_test)
            precision, recall, F1, auc = evaluate(predictions, label_test, 2)
            accuracy1, accuracy2 = evaluate(predictions, label_test, 1)
            precisionSum += precision
            recallSum += recall
            F1Sum += F1
            aucSum += auc
            R2Sum += r2_score(label_test, predictions)
            MSESum += mean_squared_error(label_test, predictions)
            accuracy1Sum += accuracy1
            accuracy2Sum += accuracy2

        outputF1 = F1Sum / 5
        outputPrecision = precisionSum / 5
        outputRecall = recallSum / 5
        outputR2 = R2Sum / 5
        outputMSE = MSESum / 5
        outputAUC = aucSum / 5
        outputAccuracy1 = accuracy1Sum / 5
        outputAccuracy2 = accuracy2Sum / 5
        '''
        if (outputRecall + outputPrecision) == 0:
            outputF1 = 0.0
        else:
            outputF1 = 2 * outputRecall * outputPrecision / (outputRecall + outputPrecision)
        '''
        print outputPrecision
        print outputRecall
        print outputF1
        print outputAUC
        print outputR2
        print outputMSE
        print outputAccuracy1
        print outputAccuracy2

        print ''
        resultFile.write(str(outputPrecision) + '\n')
        resultFile.write(str(outputRecall) + '\n')
        resultFile.write(str(outputF1) + '\n')
        resultFile.write(str(outputAUC) + '\n')
        resultFile.write(str(outputR2) + '\n')
        resultFile.write(str(outputMSE) + '\n')
        resultFile.write(str(outputAccuracy1) + '\n')
        resultFile.write(str(outputAccuracy2) + '\n')
        resultFile.write('\n')
        resultFile.flush()

    resultFile.close()


def scoreToBinary(inputList):
    outputList = []
    for item in inputList:
        if item > 7.0:
            outputList.append(1)
        else:
            outputList.append(0)

    return outputList


def evaluate(predictions, test_labels, mode):
    if len(predictions) != len(test_labels):
        print 'prediction error!'
        return 404
    if mode == 1:
        total = 0.0
        correct1 = 0.0
        correct2 = 0.0
        for index, label in enumerate(predictions):
            total += 1.0
            if round(label) == test_labels[index]:
                correct1 += 1.0
            if label - 1 <= test_labels[index] <= label + 1:
                correct2 += 1.0
        return correct1 / total, correct2 / total
    else:
        total = 0.0
        correct = 0.0
        for index, label in enumerate(predictions):
            if label == 1:
                if test_labels[index] == 1:
                    correct += 1
                total += 1
        if total == 0:
            precision = 0
        else:
            precision = correct / total
        recall = correct / test_labels.count(1)
        if (recall + precision) == 0:
            F1 = 0.0
        else:
            F1 = 2 * recall * precision / (recall + precision)
        auc = roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc


def writeLabels(groupSize, groupTitle, labelMode):
    for group in range(groupSize):
        inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_' + str(group) + '.labeled', 'r')
        labels = []
        for line in inputFile:
            item = json.loads(line.strip())
            if labelMode == 1:
                labels.append(item['label'])
            else:
                if item['label'] > 7:
                    labels.append(1)
                else:
                    labels.append(0)
        roundNum = 0
        kf = KFold(5)
        for train_indices, test_indices in kf.split(labels):
            trainLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.train', 'w')
            testLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.test', 'w')
            for train_index in train_indices:
                trainLabelFile.write(str(labels[train_index])+'\n')
            for test_index in test_indices:
                testLabelFile.write(str(labels[test_index]) + '\n')
            roundNum += 1
            trainLabelFile.close()
            testLabelFile.close()

if __name__ == "__main__":
    # [vectorMode] 1: tfidf, 2: binaryCount, 3:LDA dist
    # [featureMode] 0: content only, 1: ngram only, 2: embedding only, 3: embedding and semantic, 4: content and ngram
    # [ablation analysis] 9: all features

    #runModel(1, 'totalGroup', 2, 0, 'MaxEnt', 2, True)
    #runModel(1, 'totalGroup', 2, 0, 'MaxEnt', 2, True)

    runModel(1, 'totalGroup', 4, 0, 'SVM', 1, True, 9)
    runModel(1, 'totalGroup', 4, 0, 'MaxEnt', 1, True, 9)
    runModel(1, 'totalGroup', 4, 0, 'NaiveBayes', 1, True, 9)

    #runModel(1, 'totalGroup', 2, 0, 'SVM', 2, True)
    # runModel(5, 'simGroup', 2, 1, 'Pass')
    # runModel(5, 'topicGroup', 2, 1, 'Pass')

    # runModel(1, 'totalGroup', 2, 4, 'Pass')
    # runModel(3, 'brandGroup', 2, 4, 'Pass')
    # runModel(5, 'simGroup', 2, 4, 'Pass')
    # runModel(5, 'topicGroup', 2, 4, 'Pass')

    #writeLabels(1, 'totalGroup', 1)
