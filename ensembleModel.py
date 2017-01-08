import json
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

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

def runModel(modelList, groupTitle, featureMode, trainModel):
    precisionSum = 0.0
    recallSum = 0.0
    F1Sum = 0.0
    aucSum = 0.0
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
        trainSize = len(modelTrainData[modelList[0]])
        testSize = len(modelTestData[modelList[0]])
        for num in range(trainSize):
            temp = []
            for model in modelList:
                temp.append(modelTrainData[model][str(num)]['0'])
                temp.append(modelTrainData[model][str(num)]['1'])
            trainFeatures.append(temp)
        for num in range(testSize):
            temp = []
            for model in modelList:
                temp.append(modelTestData[model][str(num)]['0'])
                temp.append(modelTestData[model][str(num)]['1'])
            testFeatures.append(temp)

        if trainModel == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainModel == 'Pass':
            model = linear_model.PassiveAggressiveRegressor()
        elif trainModel == 'SVM':
            model = svm.SVC(probability=True)
        else:
            model = MLPClassifier(activation='logistic', learning_rate_init=0.02, learning_rate='constant',
                                  batch_size=50)

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

if __name__ == "__main__":
    runModel(['SVM', 'MaxEnt', 'MLP'], 'totalGroup', 0, 1)