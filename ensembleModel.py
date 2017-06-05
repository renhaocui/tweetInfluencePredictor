import json
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import sklearn.metrics
from sklearn.feature_extraction.text import *
import numpy as np
from scipy.sparse import hstack, csr_matrix, vstack


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


def combProbs(inputDict, splitNum):
    outputDict = {}
    for lineNum, value in inputDict.items():
        tempDict = {'0': 0.0, '1': 0.0}
        for label, prob in value.items():
            if int(label) <= splitNum:
                tempDict['0'] += prob
            else:
                tempDict['1'] += prob
        outputDict[lineNum] = tempDict
    return outputDict


def combProbs2(inputList):
    outputList = []
    for item in inputList:
        if item < 6:
            outputList.append(0)
        else:
            outputList.append(1)

    return outputList


def tenToBinary(score):
    if score > 5:
        return 1
    else:
        return 0



def dataAligner():
    dataMapperBinary = {}
    dataMapperTen = {}
    binaryInputFile = open('dataset/experiment/groups/totalGroup_binary/data_0.labeled', 'r')
    tenInputFile = open('dataset/experiment/groups/totalGroup_ten/data_0.labeled', 'r')
    for lineNum, line in enumerate(binaryInputFile):
        tweetID = json.loads(line.strip())['id']
        dataMapperBinary[tweetID] = lineNum
    binaryInputFile.close()
    for lineNum, line in enumerate(tenInputFile):
        tweetID = json.loads(line.strip())['id']
        dataMapperTen[lineNum] = tweetID
    tenInputFile.close()

    for roundNum in range(5):
        print 'Round: '+str(roundNum)
        tenIndexFileTrain = open('outputs/totalGroup_ten/Index/' + str(roundNum) + '.train', 'r')
        tenIndexFileTest = open('outputs/totalGroup_ten/Index/' + str(roundNum) + '.test', 'r')
        binaryIndexFileTrain = open('outputs/totalGroup_binary/Index/' + str(roundNum) + '.train', 'r')
        binaryIndexFileTest = open('outputs/totalGroup_binary/Index/' + str(roundNum) + '.test', 'r')
        outputBinaryIndexFileTrain = open('outputs/totalGroup_binary_aligned/Index/' + str(roundNum) + '.train', 'w')
        outputBinaryIndexFileTest = open('outputs/totalGroup_binary_aligned/Index/' + str(roundNum) + '.test', 'w')

        alignedBinaryIndexListTrain = []
        alignedBinaryIndexListTest = []
        for line in tenIndexFileTrain:
            index = int(line.strip())
            binaryIndex = dataMapperBinary[dataMapperTen[index]]
            alignedBinaryIndexListTrain.append(str(binaryIndex))
            outputBinaryIndexFileTrain.write(str(binaryIndex)+'\n')
        for line in tenIndexFileTest:
            index = int(line.strip())
            binaryIndex = dataMapperBinary[dataMapperTen[index]]
            alignedBinaryIndexListTest.append(str(binaryIndex))
            outputBinaryIndexFileTest.write(str(binaryIndex)+'\n')

        indexMapperTrain = {}
        indexMapperTest = {}
        for lineNum, line in enumerate(binaryIndexFileTrain):
            indexMapperTrain[line.strip()] = str(lineNum)
        for lineNum, line in enumerate(binaryIndexFileTest):
            indexMapperTest[line.strip()] = str(lineNum)
        binaryIndexFileTrain.close()
        binaryIndexFileTest.close()

        for model in ['SVM', 'MLP', 'MaxEnt']:
            print model
            for featureMode in ['0', '4']:
                print featureMode
                distMapper = {}
                distFileBinaryTrain = open('outputs/totalGroup_binary/'+model+'/'+featureMode+'.'+str(roundNum)+'.train', 'r')
                distFileBinaryTest = open('outputs/totalGroup_binary/'+model+'/'+featureMode+'.'+str(roundNum)+'.test', 'r')
                outputDistFileBinaryTrain = open('outputs/totalGroup_binary_aligned/'+model+'/'+featureMode+'.'+str(roundNum)+'.train', 'w')
                outputDistFileBinaryTest = open('outputs/totalGroup_binary_aligned/'+model+'/'+featureMode+'.'+str(roundNum)+'.test', 'w')

                for line in distFileBinaryTrain:
                    data = json.loads(line.strip())
                    for index, num in indexMapperTrain.items():
                        distMapper[index] = data[num]
                for line in distFileBinaryTest:
                    data = json.loads(line.strip())
                    for index, num in indexMapperTest.items():
                        distMapper[index] = data[num]
                distFileBinaryTrain.close()
                distFileBinaryTest.close()

                temp = {}
                for num, index in enumerate(alignedBinaryIndexListTrain):
                    temp[str(num)] = distMapper[index]
                outputDistFileBinaryTrain.write(json.dumps(temp)+'\n')
                temp = {}
                for num, index in enumerate(alignedBinaryIndexListTest):
                    temp[str(num)] = distMapper[index]
                outputDistFileBinaryTest.write(json.dumps(temp) + '\n')
                outputDistFileBinaryTrain.close()
                outputDistFileBinaryTest.close()

        tenIndexFileTest.close()
        tenIndexFileTrain.close()
        outputBinaryIndexFileTrain.close()
        outputBinaryIndexFileTest.close()


def runModel(modelList, groupTitle, featureMode, trainModel, evaluateMode, context=False):
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
        train_Labels = []
        test_Labels = []
        modelTrainData = {}
        modelTestData = {}

        trainLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.train', 'r')
        testLabelFile = open('outputs/' + groupTitle + '/Labels/' + str(roundNum) + '.test', 'r')
        for line in trainLabelFile:
            train_Labels.append(int(line.strip()))
        for line in testLabelFile:
            test_Labels.append(int(line.strip()))
        trainLabelFile.close()
        testLabelFile.close()
        trainLabels = combProbs2(train_Labels)
        testLabels = combProbs2(test_Labels)

        for model in modelList:
            inputTrainFile = open('outputs/' + groupTitle + '/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'r')
            for line in inputTrainFile:
                modelTrainData[model] = json.loads(line.strip())
            inputTrainFile.close()
            inputTestFile = open('outputs/' + groupTitle + '/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'r')
            for line in inputTestFile:
                modelTestData[model] = json.loads(line.strip())
            inputTestFile.close()

        trainFeatures = []
        testFeatures = []
        contentFeaturesTrain = []
        contentFeaturesTest = []
        trainSize = len(modelTrainData[modelList[0]])
        testSize = len(modelTestData[modelList[0]])
        trainData = {}
        testData = {}
        if evaluateMode == 2:
            for model in modelList:
                trainData[model] = combProbs(modelTrainData[model])
                testData[model] = combProbs(modelTestData[model])
        else:
            trainData = modelTrainData
            testData = modelTestData
        for num in range(trainSize):
            temp = []
            for model in modelList:
                temp.append(trainData[model][str(num)]['0'])
                temp.append(trainData[model][str(num)]['1'])
            trainFeatures.append(temp)
            contentFeaturesTrain.append(np.array(temp))
        for num in range(testSize):
            temp = []
            for model in modelList:
                temp.append(testData[model][str(num)]['0'])
                temp.append(testData[model][str(num)]['1'])
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
            model = MLPClassifier(activation='logistic', learning_rate='constant')

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
    outputFile.write(str(modelList)+'\t'+str(context)+'\t'+str(featureMode)+'\t'+str(evaluateMode)+'\n')
    outputFile.write(str(precisionSum/5)+'\n')
    outputFile.write(str(recallSum/5)+'\n')
    outputFile.write(str(F1Sum/5)+'\n')
    outputFile.write(str(aucSum/5)+'\n')
    outputFile.write('\n')
    outputFile.close()


def runModel2(modelList, featureMode, trainModel, context=False):
    precisionSum = 0.0
    recallSum = 0.0
    F1Sum = 0.0
    aucSum = 0.0
    contents = []
    if context:
        inputFile = open('dataset/experiment/groups/totalGroup_ten/data_0.labeled', 'r')
        for line in inputFile:
            item = json.loads(line.strip())
            contents.append(item['content'])
        inputFile.close()
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)

    for roundNum in range(5):
        print 'Round: '+str(roundNum)
        train_Labels = []
        test_Labels = []
        modelTrainDataTen = {}
        modelTrainDataBinary = {}
        modelTestDataTen = {}
        modelTestDataBinary = {}

        trainLabelFile = open('outputs/totalGroup_ten/Labels/' + str(roundNum) + '.train', 'r')
        testLabelFile = open('outputs/totalGroup_ten/Labels/' + str(roundNum) + '.test', 'r')
        for line in trainLabelFile:
            train_Labels.append(int(line.strip()))
        for line in testLabelFile:
            test_Labels.append(int(line.strip()))
        trainLabelFile.close()
        testLabelFile.close()
        trainLabels = combProbs2(train_Labels)
        testLabels = combProbs2(test_Labels)

        for model in modelList:
            inputTrainFileTen = open('outputs/totalGroup_ten/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'r')
            for line in inputTrainFileTen:
                modelTrainDataTen[model] = json.loads(line.strip())
            inputTrainFileTen.close()
            inputTrainFileBinary = open('outputs/totalGroup_binary_aligned/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'r')
            for line in inputTrainFileBinary:
                modelTrainDataBinary[model] = json.loads(line.strip())
            inputTrainFileBinary.close()

            inputTestFileTen = open('outputs/totalGroup_ten/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'r')
            for line in inputTestFileTen:
                modelTestDataTen[model] = json.loads(line.strip())
            inputTestFileTen.close()
            inputTestFileBinary = open('outputs/totalGroup_binary_aligned/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'r')
            for line in inputTestFileBinary:
                modelTestDataBinary[model] = json.loads(line.strip())
            inputTestFileBinary.close()


        trainFeatures = []
        testFeatures = []
        contentFeaturesTrain = []
        contentFeaturesTest = []
        trainSize = len(modelTrainDataTen[modelList[0]])
        testSize = len(modelTestDataTen[modelList[0]])
        trainDataTen = {}
        testDataTen = {}
        for model in modelList:
            trainDataTen[model] = combProbs(modelTrainDataTen[model])
            testDataTen[model] = combProbs(modelTestDataTen[model])
        trainDataBinary = modelTrainDataBinary
        testDataBinary = modelTestDataBinary

        for num in range(trainSize):
            temp = []
            for model in modelList:
                temp.append(trainDataTen[model][str(num)]['0'])
                temp.append(trainDataTen[model][str(num)]['1'])
                temp.append(trainDataBinary[model][str(num)]['0'])
                temp.append(trainDataBinary[model][str(num)]['1'])
            trainFeatures.append(temp)
            contentFeaturesTrain.append(np.array(temp))
        for num in range(testSize):
            temp = []
            for model in modelList:
                temp.append(testDataTen[model][str(num)]['0'])
                temp.append(testDataTen[model][str(num)]['1'])
                temp.append(testDataBinary[model][str(num)]['0'])
                temp.append(testDataBinary[model][str(num)]['1'])
            testFeatures.append(temp)
            contentFeaturesTest.append(np.array(temp))

        if context:
            print 'Generating context features...'
            trainTemp = []
            testTemp = []
            trainIndexFile = open('outputs/totalGroup_ten/Index/' + str(roundNum) + '.train', 'r')
            testIndexFile = open('outputs/totalGroup_ten/Index/' + str(roundNum) + '.test', 'r')
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
            model = MLPClassifier(activation='logistic', learning_rate='constant')

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

    outputFile = open('results/ensembleBinaryTen.'+trainModel, 'a')
    outputFile.write(str(modelList)+'\t'+str(context)+'\t'+str(featureMode)+'\n')
    outputFile.write(str(precisionSum/5)+'\n')
    outputFile.write(str(recallSum/5)+'\n')
    outputFile.write(str(F1Sum/5)+'\n')
    outputFile.write(str(aucSum/5)+'\n')
    outputFile.write('\n')
    outputFile.close()


def runModelAcrossLabel(modelList, labelMode, featureMode, ensembleFeatureMode, trainModel, splitNum, context=False):
    precisionSum = 0.0
    recallSum = 0.0
    F1Sum = 0.0
    aucSum = 0.0
    contents = []
    IDs = []
    labelData = {}

    print 'Data loading...'
    inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        contents.append(item['content'])
        IDs.append(str(item['id']))
        if item['label'] > splitNum:
            label = 1
        else:
            label = 0
        labelData[str(item['id'])] = label

    inputFile.close()

    if context:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)
        vectorData ={}
        for index, tweetID in enumerate(IDs):
            vectorData[tweetID] = vectorMatrix.getrow(index)

    for roundNum in range(5):
        print 'Round: ' + str(roundNum)
        indexFile = open('dataset/experiment/groups/totalGroup10/indices/' + str(roundNum) + '.indices', 'r')
        for line in indexFile:
            temp = json.loads(line.strip())
            trainIDs = temp['train']
            testIDs = temp['test']
        indexFile.close()

        modelDataTen = {}
        modelDataBinary = {}

        for model in modelList:
            modelDataTen[model] = {}
            modelDataBinary[model] = {}
            inputTrainFileTen = open(
                'outputs/totalGroup10_'+str(splitNum)+'/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.train', 'r')
            for line in inputTrainFileTen:
                modelDataTen[model] = dict(modelDataTen[model].items() + json.loads(line.strip()).items())
            inputTrainFileTen.close()
            inputTrainFileBinary = open(
                'outputs/totalGroup2_'+str(splitNum)+'/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.train',
                'r')
            for line in inputTrainFileBinary:
                modelDataBinary[model] = dict(modelDataBinary[model].items() + json.loads(line.strip()).items())
            inputTrainFileBinary.close()

            inputTestFileTen = open(
                'outputs/totalGroup10_'+str(splitNum)+'/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.test', 'r')
            for line in inputTestFileTen:
                modelDataTen[model] = dict(modelDataTen[model].items() + json.loads(line.strip()).items())
            inputTestFileTen.close()
            inputTestFileBinary = open(
                'outputs/totalGroup2_'+str(splitNum)+'/' + model + '/' + str(featureMode) + '.' + str(roundNum) + '.test',
                'r')
            for line in inputTestFileBinary:
                modelDataBinary[model] = dict(modelDataBinary[model].items() + json.loads(line.strip()).items())
            inputTestFileBinary.close()

        trainFeatures = []
        testFeatures = []
        contentFeaturesTrain = []
        contentFeaturesTest = []
        trainLabels = []
        testLabels = []
        dataTen = {}
        if ensembleFeatureMode == 2:
            for model in modelList:
                dataTen[model] = combProbs(modelDataTen[model], splitNum)
        else:
            dataTen = modelDataTen
        dataBinary = modelDataBinary

        for tweetID in trainIDs:
            trainLabels.append(labelData[tweetID])
            temp = []
            for model in modelList:
                if ensembleFeatureMode == 2:
                    temp.append(dataTen[model][tweetID]['0'])
                    temp.append(dataTen[model][tweetID]['1'])
                else:
                    for labelIndex in range(10):
                      temp.append(dataTen[model][tweetID][str(labelIndex+1)])
                temp.append(dataBinary[model][tweetID]['0'])
                temp.append(dataBinary[model][tweetID]['1'])
            trainFeatures.append(temp)
            contentFeaturesTrain.append(np.array(temp))
        for tweetID in testIDs:
            testLabels.append(labelData[tweetID])
            temp = []
            for model in modelList:
                if ensembleFeatureMode == 2:
                    temp.append(dataTen[model][tweetID]['0'])
                    temp.append(dataTen[model][tweetID]['1'])
                else:
                    for labelIndex in range(10):
                      temp.append(dataTen[model][tweetID][str(labelIndex+1)])
                temp.append(dataBinary[model][tweetID]['0'])
                temp.append(dataBinary[model][tweetID]['1'])
            testFeatures.append(temp)
            contentFeaturesTest.append(np.array(temp))

        if context:
            print 'Generating context features...'

            contextFeaturesTrain = vectorData[trainIDs[0]]
            contextFeaturesTest = vectorData[testIDs[0]]
            for j, tweetID in enumerate(trainIDs):
                if j > 0:
                    contextFeaturesTrain = vstack((contextFeaturesTrain, vectorData[tweetID]), format='csr')
            for j, tweetID in enumerate(testIDs):
                if j > 0:
                    contextFeaturesTest = vstack((contextFeaturesTest, vectorData[tweetID]), format='csr')
            trainFeatures = hstack((contextFeaturesTrain, csr_matrix(np.array(contentFeaturesTrain))), format='csr')
            testFeatures = hstack((contextFeaturesTest, csr_matrix(np.array(contentFeaturesTest))), format='csr')

        if trainModel == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainModel == 'Pass':
            model = linear_model.PassiveAggressiveRegressor()
        elif trainModel == 'SVM':
            model = svm.SVC()
        else:
            model = MLPClassifier(activation='logistic', learning_rate='constant')

        print 'Training...'
        model.fit(trainFeatures, trainLabels)
        print 'Inference...'
        predictions = model.predict(testFeatures)
        precision, recall, F1, auc = evaluate(predictions, testLabels, 2)
        precisionSum += precision
        recallSum += recall
        F1Sum += F1
        aucSum += auc

    print 'Precision: ' + str(precisionSum / 5)
    print 'Recall: ' + str(recallSum / 5)
    print 'F1: ' + str(F1Sum / 5)
    print 'AUC: ' + str(aucSum / 5)

    outputFile = open('results/ensemble/split'+str(splitNum)+'/ensembleAcross.' + trainModel, 'a')
    outputFile.write(str(modelList) + '\t' + str(context) + '\t' + str(featureMode) + '\t' + str(labelMode) + '\t' + str(ensembleFeatureMode) +'\n')
    outputFile.write(str(precisionSum / 5) + '\n')
    outputFile.write(str(recallSum / 5) + '\n')
    outputFile.write(str(F1Sum / 5) + '\n')
    outputFile.write(str(aucSum / 5) + '\n')
    outputFile.write('\n')
    outputFile.close()


def runModelAcrossLabelFull(modelList, labelMode, featureMode, ensembleFeatureMode, trainModel, splitNum, context=False):
    precisionSum = 0.0
    recallSum = 0.0
    F1Sum = 0.0
    aucSum = 0.0
    contents = []
    IDs = []
    labelData = {}

    print 'Data loading...'
    inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        contents.append(item['content'])
        IDs.append(str(item['id']))
        if item['label'] > splitNum:
            label = 1
        else:
            label = 0
        labelData[str(item['id'])] = label

    inputFile.close()

    if context:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)
        vectorData ={}
        for index, tweetID in enumerate(IDs):
            vectorData[tweetID] = vectorMatrix.getrow(index)

    for roundNum in range(5):
        print 'Round: ' + str(roundNum)
        indexFile = open('dataset/experiment/groups/totalGroup10/indices/' + str(roundNum) + '.indices', 'r')
        for line in indexFile:
            temp = json.loads(line.strip())
            trainIDs = temp['train']
            testIDs = temp['test']
        indexFile.close()

        modelDataTen = {}
        modelDataBinary = {}

        for model in modelList:
            inputFileTen = open(
                'outputs/totalGroup10_5_full/' + model + '/' + str(featureMode) + '.prob', 'r')
            for line in inputFileTen:
                modelDataTen[model] = json.loads(line.strip())
            inputFileTen.close()
            inputFileBinary = open(
                'outputs/totalGroup2_5_full/' + model + '/' + str(featureMode) + '.prob', 'r')
            for line in inputFileBinary:
                modelDataBinary[model] = json.loads(line.strip())
            inputFileBinary.close()

        trainFeatures = []
        testFeatures = []
        contentFeaturesTrain = []
        contentFeaturesTest = []
        trainLabels = []
        testLabels = []
        dataTen = {}
        if ensembleFeatureMode == 2:
            for model in modelList:
                dataTen[model] = combProbs(modelDataTen[model], splitNum)
        else:
            dataTen = modelDataTen
        dataBinary = modelDataBinary

        for tweetID in trainIDs:
            trainLabels.append(labelData[tweetID])
            temp = []
            for model in modelList:
                if ensembleFeatureMode == 2:
                    temp.append(dataTen[model][tweetID]['0'])
                    temp.append(dataTen[model][tweetID]['1'])
                else:
                    for labelIndex in range(10):
                      temp.append(dataTen[model][tweetID][str(labelIndex+1)])
                temp.append(dataBinary[model][tweetID]['0'])
                temp.append(dataBinary[model][tweetID]['1'])
            trainFeatures.append(temp)
            contentFeaturesTrain.append(np.array(temp))
        for tweetID in testIDs:
            testLabels.append(labelData[tweetID])
            temp = []
            for model in modelList:
                if ensembleFeatureMode == 2:
                    temp.append(dataTen[model][tweetID]['0'])
                    temp.append(dataTen[model][tweetID]['1'])
                else:
                    for labelIndex in range(10):
                      temp.append(dataTen[model][tweetID][str(labelIndex+1)])
                temp.append(dataBinary[model][tweetID]['0'])
                temp.append(dataBinary[model][tweetID]['1'])
            testFeatures.append(temp)
            contentFeaturesTest.append(np.array(temp))

        if context:
            print 'Generating context features...'

            contextFeaturesTrain = vectorData[trainIDs[0]]
            contextFeaturesTest = vectorData[testIDs[0]]
            for j, tweetID in enumerate(trainIDs):
                if j > 0:
                    contextFeaturesTrain = vstack((contextFeaturesTrain, vectorData[tweetID]), format='csr')
            for j, tweetID in enumerate(testIDs):
                if j > 0:
                    contextFeaturesTest = vstack((contextFeaturesTest, vectorData[tweetID]), format='csr')
            trainFeatures = hstack((contextFeaturesTrain, csr_matrix(np.array(contentFeaturesTrain))), format='csr')
            testFeatures = hstack((contextFeaturesTest, csr_matrix(np.array(contentFeaturesTest))), format='csr')

        if trainModel == 'MaxEnt':
            model = linear_model.LogisticRegression()
        elif trainModel == 'Pass':
            model = linear_model.PassiveAggressiveRegressor()
        elif trainModel == 'SVM':
            model = svm.SVC()
        else:
            model = MLPClassifier(activation='logistic', learning_rate='constant')

        print 'Training...'
        model.fit(trainFeatures, trainLabels)
        print 'Inference...'
        predictions = model.predict(testFeatures)
        precision, recall, F1, auc = evaluate(predictions, testLabels, 2)
        precisionSum += precision
        recallSum += recall
        F1Sum += F1
        aucSum += auc

    print 'Precision: ' + str(precisionSum / 5)
    print 'Recall: ' + str(recallSum / 5)
    print 'F1: ' + str(F1Sum / 5)
    print 'AUC: ' + str(aucSum / 5)

    outputFile = open('results/ensemble/split5_full/ensembleAcross.' + trainModel, 'a')
    outputFile.write(str(modelList) + '\t' + str(context) + '\t' + str(featureMode) + '\t' + str(labelMode) + '\t' + str(ensembleFeatureMode) +'\n')
    outputFile.write(str(precisionSum / 5) + '\n')
    outputFile.write(str(recallSum / 5) + '\n')
    outputFile.write(str(F1Sum / 5) + '\n')
    outputFile.write(str(aucSum / 5) + '\n')
    outputFile.write('\n')
    outputFile.close()


def runModelAcrossLabelHandout(modelList, labelMode, featureMode, ensembleFeatureMode, trainModel, splitNum, context=False):
    contents = []
    IDs = []
    labelData = {}

    print 'Data loading...'
    inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        contents.append(item['content'])
        IDs.append(str(item['id']))
        if labelMode == 1:
            labelData[str(item['id'])] = item['label']
        else:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
            labelData[str(item['id'])] = label

    inputFile.close()

    if context:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)
        vectorData ={}
        for index, tweetID in enumerate(IDs):
            vectorData[tweetID] = vectorMatrix.getrow(index)

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    modelDataTrainTen = {}
    modelDataTrainBinary = {}
    modelDataTestTen = {}
    modelDataTestBinary = {}

    for model in modelList:
        inputFileTrainTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/' + str(featureMode) + '.train', 'r')
        for line in inputFileTrainTen:
            modelDataTrainTen[model] = json.loads(line.strip())
        inputFileTrainTen.close()
        inputFileTrainBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/' + str(featureMode) + '.train', 'r')
        for line in inputFileTrainBinary:
            modelDataTrainBinary[model] = json.loads(line.strip())
        inputFileTrainBinary.close()

        inputFileTestTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/' + str(featureMode) + '.test', 'r')
        for line in inputFileTestTen:
            modelDataTestTen[model] = json.loads(line.strip())
        inputFileTestTen.close()
        inputFileTestBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/' + str(featureMode) + '.test', 'r')
        for line in inputFileTestBinary:
            modelDataTestBinary[model] = json.loads(line.strip())
        inputFileTestBinary.close()

    trainFeatures = []
    testFeatures = []
    contentFeaturesTrain = []
    contentFeaturesTest = []
    trainLabels = []
    testLabels = []
    dataTrainTen = {}
    dataTestTen = {}
    if ensembleFeatureMode == 2:
        for model in modelList:
            dataTrainTen[model] = combProbs(modelDataTrainTen[model], splitNum)
            dataTestTen[model] = combProbs(modelDataTestTen[model], splitNum)
    else:
        dataTrainTen = modelDataTrainTen
        dataTestTen = modelDataTestTen
    dataTrainBinary = modelDataTrainBinary
    dataTestBinary = modelDataTestBinary

    for tweetID in trainIDs:
        trainLabels.append(labelData[tweetID])
        temp = []
        for model in modelList:
            if ensembleFeatureMode == 2:
                temp.append(dataTrainTen[model][tweetID]['0'])
                temp.append(dataTrainTen[model][tweetID]['1'])
            else:
                for labelIndex in range(10):
                  temp.append(dataTrainTen[model][tweetID][str(labelIndex+1)])
            temp.append(dataTrainBinary[model][tweetID]['0'])
            temp.append(dataTrainBinary[model][tweetID]['1'])
        trainFeatures.append(temp)
        contentFeaturesTrain.append(np.array(temp))

    for tweetID in testIDs:
        testLabels.append(labelData[tweetID])
        temp = []
        for model in modelList:
            if ensembleFeatureMode == 2:
                temp.append(dataTestTen[model][tweetID]['0'])
                temp.append(dataTestTen[model][tweetID]['1'])
            else:
                for labelIndex in range(10):
                  temp.append(dataTestTen[model][tweetID][str(labelIndex+1)])
            temp.append(dataTestBinary[model][tweetID]['0'])
            temp.append(dataTestBinary[model][tweetID]['1'])
        testFeatures.append(temp)
        contentFeaturesTest.append(np.array(temp))

    if context:
        print 'Generating context features...'

        contextFeaturesTrain = vectorData[trainIDs[0]]
        contextFeaturesTest = vectorData[testIDs[0]]
        for j, tweetID in enumerate(trainIDs):
            if j > 0:
                contextFeaturesTrain = vstack((contextFeaturesTrain, vectorData[tweetID]), format='csr')
        for j, tweetID in enumerate(testIDs):
            if j > 0:
                contextFeaturesTest = vstack((contextFeaturesTest, vectorData[tweetID]), format='csr')
        trainFeatures = hstack((contextFeaturesTrain, csr_matrix(np.array(contentFeaturesTrain))), format='csr')
        testFeatures = hstack((contextFeaturesTest, csr_matrix(np.array(contentFeaturesTest))), format='csr')

    if trainModel == 'MaxEnt':
        model = linear_model.LogisticRegression()
    elif trainModel == 'Pass':
        model = linear_model.PassiveAggressiveRegressor()
    elif trainModel == 'SVM':
        model = svm.SVC()
    else:
        model = MLPClassifier(activation='logistic', learning_rate='constant')

    print 'Training...'
    model.fit(trainFeatures, trainLabels)
    print 'Inference...'
    predictions = model.predict(testFeatures)
    precision, recall, F1, auc = evaluate(predictions, testLabels, labelMode, splitNum)
    precisionSum = precision
    recallSum = recall
    F1Sum = F1
    aucSum = auc

    print 'Precision: ' + str(precisionSum)
    print 'Recall: ' + str(recallSum)
    print 'F1: ' + str(F1Sum)
    print 'AUC: ' + str(aucSum)

    outputFile = open('results/ensemble/split5_handout/ensembleAcross.' + trainModel, 'a')
    outputFile.write(str(modelList) + '\t' + str(context) + '\t' + str(featureMode) + '\t' + str(labelMode) + '\t' + str(ensembleFeatureMode) +'\n')
    outputFile.write(str(precisionSum) + '\n')
    outputFile.write(str(recallSum) + '\n')
    outputFile.write(str(F1Sum) + '\n')
    outputFile.write(str(aucSum) + '\n')
    outputFile.write('\n')
    outputFile.close()


def runModelAcrossLabelHandout2(modelList, labelMode, ensembleFeatureMode, trainModel, splitNum, context=False):
    contents = []
    IDs = []
    labelData = {}

    print 'Data loading...'
    inputFile = open('dataset/experiment/groups/totalGroup/data_0.labeled', 'r')
    for line in inputFile:
        item = json.loads(line.strip())
        contents.append(item['content'])
        IDs.append(str(item['id']))
        if labelMode == 1:
            labelData[str(item['id'])] = item['label']
        else:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
            labelData[str(item['id'])] = label

    inputFile.close()

    if context:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english', binary='True')
        vectorMatrix = vectorizer.fit_transform(contents)
        vectorData ={}
        for index, tweetID in enumerate(IDs):
            vectorData[tweetID] = vectorMatrix.getrow(index)

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    modelDataTrainTen0 = {}
    modelDataTrainBinary0 = {}
    modelDataTestTen0 = {}
    modelDataTestBinary0 = {}

    for model in modelList:
        inputFileTrainTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/0.train', 'r')
        for line in inputFileTrainTen:
            modelDataTrainTen0[model] = json.loads(line.strip())
        inputFileTrainTen.close()
        inputFileTrainBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/0.train', 'r')
        for line in inputFileTrainBinary:
            modelDataTrainBinary0[model] = json.loads(line.strip())
        inputFileTrainBinary.close()

        inputFileTestTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/0.test', 'r')
        for line in inputFileTestTen:
            modelDataTestTen0[model] = json.loads(line.strip())
        inputFileTestTen.close()
        inputFileTestBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/0.test', 'r')
        for line in inputFileTestBinary:
            modelDataTestBinary0[model] = json.loads(line.strip())
        inputFileTestBinary.close()


    modelDataTrainTen1 = {}
    modelDataTrainBinary1 = {}
    modelDataTestTen1 = {}
    modelDataTestBinary1 = {}

    for model in modelList:
        inputFileTrainTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/1.train', 'r')
        for line in inputFileTrainTen:
            modelDataTrainTen1[model] = json.loads(line.strip())
        inputFileTrainTen.close()
        inputFileTrainBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/1.train', 'r')
        for line in inputFileTrainBinary:
            modelDataTrainBinary1[model] = json.loads(line.strip())
        inputFileTrainBinary.close()

        inputFileTestTen = open(
            'outputs/totalGroup10_5_handout/' + model + '/1.test', 'r')
        for line in inputFileTestTen:
            modelDataTestTen1[model] = json.loads(line.strip())
        inputFileTestTen.close()
        inputFileTestBinary = open(
            'outputs/totalGroup2_5_handout/' + model + '/1.test', 'r')
        for line in inputFileTestBinary:
            modelDataTestBinary1[model] = json.loads(line.strip())
        inputFileTestBinary.close()


    dataTrainTen0 = {}
    dataTestTen0 = {}
    dataTrainTen1 = {}
    dataTestTen1 = {}
    if ensembleFeatureMode == 2:
        for model in modelList:
            dataTrainTen0[model] = combProbs(modelDataTrainTen0[model], splitNum)
            dataTestTen0[model] = combProbs(modelDataTestTen0[model], splitNum)
            dataTrainTen1[model] = combProbs(modelDataTrainTen1[model], splitNum)
            dataTestTen1[model] = combProbs(modelDataTestTen1[model], splitNum)
    else:
        dataTrainTen0 = modelDataTrainTen0
        dataTestTen0 = modelDataTestTen0
        dataTrainTen1 = modelDataTrainTen1
        dataTestTen1 = modelDataTestTen1
    dataTrainBinary0 = modelDataTrainBinary0
    dataTestBinary0 = modelDataTestBinary0
    dataTrainBinary1 = modelDataTrainBinary1
    dataTestBinary1 = modelDataTestBinary1

    trainFeatures = []
    testFeatures = []
    contentFeaturesTrain = []
    contentFeaturesTest = []
    trainLabels = []
    testLabels = []

    for tweetID in trainIDs:
        trainLabels.append(labelData[tweetID])
        temp = []
        for model in modelList:
            if ensembleFeatureMode == 2:
                temp.append(dataTrainTen0[model][tweetID]['0'])
                temp.append(dataTrainTen0[model][tweetID]['1'])
                temp.append(dataTrainTen1[model][tweetID]['0'])
                temp.append(dataTrainTen1[model][tweetID]['1'])
            else:
                for labelIndex in range(10):
                  temp.append(dataTrainTen0[model][tweetID][str(labelIndex+1)])
                  temp.append(dataTrainTen1[model][tweetID][str(labelIndex+1)])
            temp.append(dataTrainBinary0[model][tweetID]['0'])
            temp.append(dataTrainBinary0[model][tweetID]['1'])
            temp.append(dataTrainBinary1[model][tweetID]['0'])
            temp.append(dataTrainBinary1[model][tweetID]['1'])
        trainFeatures.append(temp)
        contentFeaturesTrain.append(np.array(temp))

    for tweetID in testIDs:
        testLabels.append(labelData[tweetID])
        temp = []
        for model in modelList:
            if ensembleFeatureMode == 2:
                temp.append(dataTestTen0[model][tweetID]['0'])
                temp.append(dataTestTen0[model][tweetID]['1'])
                temp.append(dataTestTen1[model][tweetID]['0'])
                temp.append(dataTestTen1[model][tweetID]['1'])
            else:
                for labelIndex in range(10):
                  temp.append(dataTestTen0[model][tweetID][str(labelIndex+1)])
                  temp.append(dataTestTen1[model][tweetID][str(labelIndex+1)])
            temp.append(dataTestBinary0[model][tweetID]['0'])
            temp.append(dataTestBinary0[model][tweetID]['1'])
            temp.append(dataTestBinary1[model][tweetID]['0'])
            temp.append(dataTestBinary1[model][tweetID]['1'])
        testFeatures.append(temp)
        contentFeaturesTest.append(np.array(temp))

    if context:
        print 'Generating context features...'

        contextFeaturesTrain = vectorData[trainIDs[0]]
        contextFeaturesTest = vectorData[testIDs[0]]
        for j, tweetID in enumerate(trainIDs):
            if j > 0:
                contextFeaturesTrain = vstack((contextFeaturesTrain, vectorData[tweetID]), format='csr')
        for j, tweetID in enumerate(testIDs):
            if j > 0:
                contextFeaturesTest = vstack((contextFeaturesTest, vectorData[tweetID]), format='csr')
        trainFeatures = hstack((contextFeaturesTrain, csr_matrix(np.array(contentFeaturesTrain))), format='csr')
        testFeatures = hstack((contextFeaturesTest, csr_matrix(np.array(contentFeaturesTest))), format='csr')

    if trainModel == 'MaxEnt':
        model = linear_model.LogisticRegression()
    elif trainModel == 'Pass':
        model = linear_model.PassiveAggressiveRegressor()
    elif trainModel == 'SVM':
        model = svm.SVC()
    else:
        model = MLPClassifier(activation='logistic', learning_rate='constant')

    print 'Training...'
    model.fit(trainFeatures, trainLabels)
    print 'Inference...'
    predictions = model.predict(testFeatures)
    precision, recall, F1, auc = evaluate(predictions, testLabels, labelMode, splitNum)
    precisionSum = precision
    recallSum = recall
    F1Sum = F1
    aucSum = auc

    print 'Precision: ' + str(precisionSum)
    print 'Recall: ' + str(recallSum)
    print 'F1: ' + str(F1Sum)
    print 'AUC: ' + str(aucSum)

    outputFile = open('results/ensemble/split5_handout/ensembleAcross.' + trainModel, 'a')
    outputFile.write(str(modelList) + '\t' + str(context) + '\t' + str(labelMode) + '\t' + str(ensembleFeatureMode) +'\t across feature'+'\n')
    outputFile.write(str(precisionSum) + '\n')
    outputFile.write(str(recallSum) + '\n')
    outputFile.write(str(F1Sum) + '\n')
    outputFile.write(str(aucSum) + '\n')
    outputFile.write('\n')
    outputFile.close()



if __name__ == "__main__":
    groupTitle = 'totalGroup'

    # modelList, labelMode, featureMode, ensembleFeatureMode, trainModel, splitNum, context=False
    # [featureMode] 0: content only, 1: ngram only, 2: embedding only, 3: embedding and semantic, 4: content and ngram

    runModelAcrossLabelHandout(['LLDA'], 1, 1, 2, 'SVM', 5, context=False)
    runModelAcrossLabelHandout(['LLDA'], 1, 1, 1, 'SVM', 5, context=False)

    #runModelAcrossLabelHandout2(['SVM', 'MaxEnt'], 2, 2, 'SVM', 5, context=False)
    #runModelAcrossLabelHandout2(['SVM', 'LLDA'], 2, 1, 'SVM', 5, context=False)
    #runModelAcrossLabelHandout2(['SVM', 'MaxEnt'], 2, 2, 'SVM', 5, context=True)
