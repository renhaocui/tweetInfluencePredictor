import json
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import sklearn.metrics
from sklearn.feature_extraction.text import *
import numpy as np
from scipy.sparse import hstack, csr_matrix

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
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc


def runModel(modelList, groupTitle, featureMode, trainModel, context=False):
    precisionSum = 0.0
    recallSum = 0.0
    F1Sum = 0.0
    aucSum = 0.0
    contents = []
    if context:
        inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
        for line in inputFile:
            item = json.loads(line.strip())
            contents.append(item['content'])
        inputFile.close()
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    for roundNum in range(5):
        print 'Round: '+str(roundNum)
        trainLabels = []
        testLabels = []
        modelTrainData = {}
        modelTestData = {}

        trainLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.train', 'r')
        testLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.test', 'r')
        for line in trainLabelFile:
            trainLabels.append(int(line.strip()))
        for line in testLabelFile:
            testLabels.append(int(line.strip()))
        trainLabelFile.close()
        testLabelFile.close()

        for model in modelList:
            inputTrainFile = open('outputs/' + groupTitle + '/' + model + '/' + str(featureMode) + '.' + str(roundNum+1) + '.train', 'r')
            for line in inputTrainFile:
                modelTrainData[model] = json.loads(line.strip())
            inputTrainFile.close()
            inputTestFile = open('outputs/' + groupTitle + '/' + model + '/' + str(featureMode) + '.' + str(roundNum+1) + '.test', 'r')
            for line in inputTestFile:
                modelTestData[model] = json.loads(line.strip())
            inputTestFile.close()

        trainFeatures = []
        testFeatures = []
        contentFeaturesTrain = []
        contentFeaturesTest = []
        trainSize = len(modelTrainData[modelList[0]])
        testSize = len(modelTestData[modelList[0]])
        for num in range(trainSize):
            temp = []
            for model in modelList:
                temp.append(modelTrainData[model][str(num)]['0'])
                temp.append(modelTrainData[model][str(num)]['1'])
            trainFeatures.append(temp)
            contentFeaturesTrain.append(np.array(temp))
        for num in range(testSize):
            temp = []
            for model in modelList:
                temp.append(modelTestData[model][str(num)]['0'])
                temp.append(modelTestData[model][str(num)]['1'])
            testFeatures.append(temp)
            contentFeaturesTest.append(np.array(temp))

        if context:
            print 'Generating context features...'
            trainTemp = []
            testTemp = []
            trainIndexFile = open('outputs/' + groupTitle + '/Index/' + str(roundNum) + '.train', 'r')
            testIndexFile = open('outputs/' + groupTitle + '/Index/' + str(roundNum) + '.test', 'r')
            for line in trainIndexFile:
                trainTemp.append(int(line.strip()))
            for line in testIndexFile:
                testTemp.append(int(line.strip()))
            trainIndexFile.close()
            testIndexFile.close()

            trainIndex = np.array(trainTemp)
            testIndex = np.array(testTemp)
            contextFeaturesTrain = vectorMatrix[trainIndex]
            contextFeaturesTest = vectorMatrix[testIndex]
            trainFeatures = hstack((contextFeaturesTrain, csr_matrix(np.array(contentFeaturesTrain))), format='csr')
            testFeatures = hstack((contextFeaturesTest, csr_matrix(np.array(contentFeaturesTest))), format='csr')

        if trainModel == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainModel == 'Pass':
            model = linear_model.PassiveAggressiveRegressor()
        elif trainModel == 'SVM':
            model = svm.SVC()
        else:
            model = MLPClassifier(activation='logistic', learning_rate_init=0.001, learning_rate='constant')

        model.fit(trainFeatures, trainLabels)
        predictions = model.predict(testFeatures)
        precision, recall, F1, auc = evaluate(predictions, testLabels, 2)
        precisionSum += precision
        recallSum += recall
        F1Sum += F1
        aucSum += auc

    print 'Precision: '+str(precisionSum/5)
    print 'Recall: '+str(recallSum/5)
    print 'F1: '+str(F1Sum/5)
    print 'AUC: '+str(aucSum/5)

    outputFile = open('results/ensemble.'+trainModel, 'a')
    outputFile.write(str(modelList)+'\t'+str(context)+'\n')
    outputFile.write(str(precisionSum/5)+'\n')
    outputFile.write(str(recallSum/5)+'\n')
    outputFile.write(str(F1Sum/5)+'\n')
    outputFile.write(str(aucSum/5)+'\n')
    outputFile.write('\n')
    outputFile.close()

if __name__ == "__main__":
    groupTitle = 'totalGroup_content_only'

    #runModel(['SVM', 'MaxEnt', 'MLP'], groupTitle, 0, 'SVM', True)
    #runModel(['SVM', 'MaxEnt', 'MLP'], groupTitle, 0, 'MaxEnt', True)
    runModel(['SVM', 'MaxEnt', 'MLP'], groupTitle, 0, 'MLP', True)

    runModel(['SVM', 'MaxEnt'], groupTitle, 0, 'SVM', True)
    runModel(['SVM', 'MaxEnt'], groupTitle, 0, 'MaxEnt', True)
    runModel(['SVM', 'MaxEnt'], groupTitle, 0, 'MLP', True)

    runModel(['SVM', 'MLP'], groupTitle, 0, 'SVM', True)
    runModel(['SVM', 'MLP'], groupTitle, 0, 'MaxEnt', True)
    runModel(['SVM', 'MLP'], groupTitle, 0, 'MLP', True)

    runModel(['MaxEnt', 'MLP'], groupTitle, 0, 'SVM', True)
    runModel(['MaxEnt', 'MLP'], groupTitle, 0, 'MaxEnt', True)
    runModel(['MaxEnt', 'MLP'], groupTitle, 0, 'MLP', True)