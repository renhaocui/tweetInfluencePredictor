import joblib
import json
import numpy
from sklearn import svm
from sklearn import linear_model
import sklearn.metrics
from sklearn.neural_network import MLPClassifier


def scoreToBinary(inputList, splitNum):
    outputList = []
    for item in inputList:
        if item > splitNum:
            outputList.append(1)
        else:
            outputList.append(0)

    return outputList

def evaluate(predictions, test_labels, mode, splitNum):
    if len(predictions) != len(test_labels):
        print 'prediction error!'
        return 404
    if mode == 1:
        test_labels = scoreToBinary(test_labels, splitNum)
        predictions = scoreToBinary(predictions, splitNum)
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc
    else:
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc


def runModel_MIT(labelMode, splitNum):


    embData = joblib.load('models/tweet2vec/MIT/train_general_tweet.npy')

    print type(embData)
    print len(embData)
    print len(embData[0])
    print embData[0]
    '''
    trainData = joblib.load('dataset/experiment/vector/train_emd.npy')
    testData = joblib.load('dataset/experiment/vector/test_emd.npy')

    print len(trainData)
    print len(testData)
    print trainData[4323]

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    dataFile = open('dataset/experiment/clean.labeled', 'r')
    trainLabels = []
    testLabels = []
    for line in dataFile:
        item = json.loads(line.strip())
        if labelMode == 1:
            label = item['label']
        elif labelMode == 2:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
        if item['id'] in trainIDs:
            trainLabels.append(label)
        elif item['id'] in testIDs:
            testLabels.append(label)
    '''

def runModel_CMU(trainMode, labelMode, splitNum):
    print 'Loading vectors...'
    data = numpy.load('dataset/experiment/vector/CMU_total_emd.npy')
    print len(data)
    print type(data)
    #validate vector representation
    for num, item in enumerate(data):
        if len(item) < 200:
            print num
            print len(item)
            print item

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    tweetData = {}
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    for num, line in enumerate(dataFile):
        item = json.loads(line.strip())
        if labelMode == 1:
            label = item['label']
        else:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
        tweetData[str(item['id'])] = {'feature': data[num], 'label': label}
    dataFile.close()
    print len(tweetData)

    print 'Splitting data...'
    trainFeature = []
    testFeature = []
    trainLabel = []
    testLabel = []
    for tweetID in trainIDs:
        trainFeature.append(tweetData[tweetID]['feature'])
        trainLabel.append(tweetData[tweetID]['label'])
    for tweetID in testIDs:
        testFeature.append(tweetData[tweetID]['feature'])
        testLabel.append(tweetData[tweetID]['label'])

    #trainFeature = numpy.array(trainFeature).reshape(1, -1)
    #testFeature = numpy.array(testFeature).reshape(1, -1)

    if trainMode == 'MaxEnt':
        model = linear_model.LogisticRegression()
    elif trainMode == 'SVM':
        model = svm.SVC()
    else:
        model = MLPClassifier(activation='logistic', learning_rate='constant')

    print 'Training...'
    model.fit(trainFeature, trainLabel)
    print 'Inference...'
    predictions = model.predict(testFeature)
    precision, recall, F1, auc = evaluate(predictions, testLabel, labelMode, splitNum)

    print precision
    print recall
    print F1
    print auc



def runModel_CMU(trainMode, labelMode, splitNum):
    print 'Loading vectors...'
    data = numpy.load('dataset/experiment/vector/embeddings.npy')
    print len(data)
    #validate vector representation
    for num, item in enumerate(data):
        if len(item) < 200:
            print num
            print len(item)
            print item

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    tweetData = {}
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    for num, line in enumerate(dataFile):
        item = json.loads(line.strip())
        if labelMode == 1:
            label = item['label']
        else:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
        tweetData[str(item['id'])] = {'feature': data[num], 'label': label}
    dataFile.close()
    print len(tweetData)

    print 'Splitting data...'
    trainFeature = []
    testFeature = []
    trainLabel = []
    testLabel = []
    for tweetID in trainIDs:
        trainFeature.append(tweetData[tweetID]['feature'])
        trainLabel.append(tweetData[tweetID]['label'])
    for tweetID in testIDs:
        testFeature.append(tweetData[tweetID]['feature'])
        testLabel.append(tweetData[tweetID]['label'])

    #trainFeature = numpy.array(trainFeature).reshape(1, -1)
    #testFeature = numpy.array(testFeature).reshape(1, -1)

    if trainMode == 'MaxEnt':
        model = linear_model.LogisticRegression()
    elif trainMode == 'SVM':
        model = svm.SVC()
    else:
        model = MLPClassifier(activation='logistic', learning_rate='constant')

    print 'Training...'
    model.fit(trainFeature, trainLabel)
    print 'Inference...'
    predictions = model.predict(testFeature)
    precision, recall, F1, auc = evaluate(predictions, testLabel, labelMode, splitNum)

    print precision
    print recall
    print F1
    print auc

if __name__ == "__main__":
    runModel_CMU('SVM', 2, 5)
    #runModel_MIT(2, 5)