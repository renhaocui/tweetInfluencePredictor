__author__ = 'renhao.cui'
import os
import shutil
import glob
import subprocess
import json
import sklearn.metrics
import operator

def modifyFiles(fileName):
    file = open(fileName, 'r')
    fileContent = []
    for line in file:
        fileContent.append(line.replace('Labeled', ''))
    file.close()
    # os.remove(fileName)
    newFile = open(fileName, 'w')
    for line in fileContent:
        newFile.write(line)
    newFile.close()

def formLDAFile(topicList, contentList, model):
    if model == 'train':
        outputFile = open('TMT/LDAFormatTrain.csv', 'w')
    elif model == 'test':
        outputFile = open('TMT/LDAFormatTest.csv', 'w')
    size = len(contentList)
    for t in range(size):
        outputFile.write('"' + topicList[t] + '","' + contentList[t].replace('"', "'").encode('utf-8') + '"\n')
    outputFile.close()


def scoreToBinary(inputList, splitNum):
    outputList = []
    for item in inputList:
        if item > splitNum:
            outputList.append(1)
        else:
            outputList.append(0)

    return outputList


def evaluate(predictions, test_labels, mode, splitNum=5):
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
    elif mode == 2:
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc


def outputParser(mode):
    labelData = []
    distDict = {}
    labelFile = open('LLDAoutput/label-index.txt', 'r')
    for line in labelFile:
        labelData.append(line.strip())
    labelFile.close()

    if mode == 'test':
        distFile = open('LLDAoutput/LDAFormatTest-document-topic-distributuions.csv', 'r')
    else:
        distFile = open('LLDAoutput/LDAFormatTrain-document-topic-distributuions.csv', 'r')
    for line in distFile:
        temp = line.strip().split(',')
        for i in range(len(temp)):
            if i == 0:
                indexNum = int(temp[i])
                distDict[indexNum] = {}
            else:
                distDict[indexNum][labelData[i-1]] = float(temp[i])
    distFile.close()
    return distDict


def LLDARunner(groupTitle, labelMode, splitNum):
    inputFile = open('dataset/experiment/groups/' + groupTitle + '/data_0.labeled', 'r')
    labelData = {}
    contentData = {}
    for line in inputFile:
        item = json.loads(line.strip())
        contentData[str(item['id'])] = item['content']
        if labelMode == 1:
            label = item['label']
        elif labelMode == 2:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
        labelData[str(item['id'])] = str(label)
    inputFile.close()

    precision = 0.0
    recall = 0.0
    F1 = 0.0
    auc = 0.0

    for roundNum in range(1):
        print 'Round: ' + str(roundNum)
        indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
        for line in indexFile:
            temp = json.loads(line.strip())
            trainIDs = temp['train']
            testIDs = temp['test']
        indexFile.close()
        label_test = []
        label_train = []
        content_test = []
        content_train = []

        for tweetID in testIDs:
            label_test.append(labelData[str(tweetID)])
            content_test.append(contentData[str(tweetID)])
        for tweetID in trainIDs:
            label_train.append(labelData[str(tweetID)])
            content_train.append(contentData[str(tweetID)])

        formLDAFile(label_train, content_train, 'train')
        formLDAFile(label_test, content_test, 'test')

        for file in glob.glob('TMT\\LDAFormat*.csv.term-counts.cache*'):
            os.remove(file)
        if os.path.exists('TMT Snapshots'):
            shutil.rmtree('TMT Snapshots')

        print "train LLDA..."
        subprocess.check_output('java -jar TMT/tmt-0.4.0.jar TMT/train.scala', shell=True)

        print "generate LLDA test output..."
        modifyFiles('TMT Snapshots/01000/description.txt')
        modifyFiles('TMT Snapshots/01000/params.txt')
        subprocess.check_output('java -jar TMT/tmt-0.4.0.jar TMT/test.scala', shell=True)
        shutil.copy('TMT Snapshots/LDAFormatTest-document-topic-distributuions.csv', 'LLDAoutput/')
        shutil.copy('TMT Snapshots/01000/label-index.txt', 'LLDAoutput/')

        distDict = outputParser('test')
        valid_labels_test = []
        predictions = []
        probData_test = {}
        for lineNum, dist in distDict.items():
            valid_labels_test.append(int(label_test[lineNum]))
            probData_test[testIDs[lineNum]] = dist
            sorted_temp = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
            predictions.append(int(sorted_temp[0][0]))
        if len(valid_labels_test) != len(predictions):
            print 'Inference output size error!'

        print 'generate LLDA train output...'
        for file in glob.glob('TMT\\LDAFormat*.csv.term-counts.cache*'):
            os.remove(file)
        subprocess.check_output('java -jar TMT/tmt-0.4.0.jar TMT/test2.scala', shell=True)
        shutil.copy('TMT Snapshots/LDAFormatTrain-document-topic-distributuions.csv', 'LLDAoutput/')
        shutil.copy('TMT Snapshots/01000/label-index.txt', 'LLDAoutput/')

        distDict = outputParser('train')
        probData_train = {}
        for lineNum, dist in distDict.items():
            probData_train[trainIDs[lineNum]] = dist

        print str(len(trainIDs))+'\t'+str(len(probData_train))
        print str(len(testIDs))+'\t'+str(len(probData_test))

        if labelMode == 1:
            outputFileTrain = open(
                'outputs/totalGroup10_' + str(splitNum) + '_handout/LLDA/1.train', 'w')
            outputFileTest = open(
                'outputs/totalGroup10_' + str(splitNum) + '_handout/LLDA/1.test', 'w')
        else:
            outputFileTrain = open(
                'outputs/totalGroup2_' + str(splitNum) + '_handout/LLDA/1.train', 'w')
            outputFileTest = open(
                'outputs/totalGroup2_' + str(splitNum) + '_handout/LLDA/1.test', 'w')

        outputFileTest.write(json.dumps(probData_test))
        outputFileTrain.write(json.dumps(probData_train))

        outputFileTrain.close()
        outputFileTest.close()

        precision, recall, F1, auc = evaluate(predictions, valid_labels_test, labelMode, splitNum=splitNum)

    print precision/1
    print recall/1
    print F1/1
    print auc/1


if __name__ == "__main__":
    LLDARunner('totalGroup', 2, 5)