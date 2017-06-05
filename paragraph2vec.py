import gensim
import json
from tokenizer import simpleTokenize
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

def runModel(labelMode, splitNum, trainMode, loadModel=True):
    print 'loading data...'
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
        tweetData[str(item['id'])] = {'content': item['content'], 'label': label}
    dataFile.close()
    print len(tweetData)

    trainData = []
    trainContent = []
    for num, tweetID in enumerate(trainIDs):
        content = tweetData[tweetID]['content']
        words = simpleTokenize(content)
        uni_words = [unicode(word) for word in words]
        trainContent.append(words)
        trainData.append(gensim.models.doc2vec.TaggedDocument(words=uni_words, tags=str(num)))

    if loadModel:
        print 'loading model...'
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
    else:
        print 'building model...'
        model = gensim.models.doc2vec.Doc2Vec(documents=trainData, size=200, window=5, min_count=1, workers=4, iter=20)
        model.save('models/doc2vec/test.d2v')

    trainFeature = []
    trainLabel = []
    for num, tweetID in enumerate(trainIDs):
        trainFeature.append(model.infer_vector(trainContent[num]))
        trainLabel.append(tweetData[tweetID]['label'])

    testFeature = []
    testLabel = []
    for tweetID in testIDs:
        content = tweetData[tweetID]['content']
        words = simpleTokenize(content)
        testFeature.append(model.infer_vector(words))
        testLabel.append(tweetData[tweetID]['label'])

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
    runModel(2, 5, 'MaxEnt', True)
    #runModel(2, 5, 'SVM', True)