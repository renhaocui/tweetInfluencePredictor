import json
import statistics as stat
from tokenizer import simpleTokenize

def labelStat(groupMode, groupSize):
    tweetData = {}
    inputFile = open('dataset/experiment/' + groupMode + '_' + str(groupSize) + '.labeled', 'r')
    for line in inputFile:
        temp = json.loads(line.strip())

    statFile = open('dataset/stats/' + groupMode + '_' + str(groupSize) + '.state', 'w')


def analysis():
    filterTerms = ['iphone 7', 'pikachu', 'pokemon go', 'macbook pro', 'trump', 'note 7']
    ratioList = []
    resultFile = open('dataset/experiment/total_filtered.txt', 'w')
    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        #tweetID = data['id']
        text = data['text'].encode('utf-8')
        filtered = False
        for term in filterTerms:
            if term in text.lower():
                filtered = True
                break
        if not filtered:
            finalIndex = len(data['dynamic']) - 1
            retweet = float(data['dynamic'][finalIndex]['retweet_count'])
            favorite = float(data['dynamic'][finalIndex]['favorite_count'])
            #followers = float(data['dynamic'][finalIndex]['user_followers_count'])
            if retweet == 0:
                ratio = 0
            else:
                ratio = favorite / retweet
            ratioList.append(ratio)
            resultFile.write(str(favorite)+'\t'+str(retweet)+'\t'+str(ratio)+'\n')

    resultFile.close()
    inputFile.close()
    print stat.mean(ratioList)


def lengthStat():
    inputFile = open('dataset/experiment/total.json', 'r')
    lengthList = []
    for line in inputFile:
        data = json.loads(line.strip())
        text = data['text'].encode('utf-8')
        words = simpleTokenize(text)
        twLen = float(len(words))
        lengthList.append(twLen)
    inputFile.close()
    print stat.mean(lengthList)
    print stat.stdev(lengthList)


def outAnalysis(comparedModel, mode):
    outContentFile = open('outputs/analysis/content.0', 'r')
    outTrueLabelFile = open('outputs/analysis/trueLabel.0', 'r')
    outPredictLabelFile = open('outputs/analysis/predLabel.0', 'r')
    outCompareLabelFile = open('outputs/analysis/predLabel.' + str(comparedModel), 'r')

    contents = []
    for line in outContentFile:
        contents.append(line.strip())
    outContentFile.close()

    trueLabels = []
    for line in outTrueLabelFile:
        trueLabels.append(line.strip())
    outTrueLabelFile.close()

    predLabels = []
    for line in outPredictLabelFile:
        predLabels.append(line.strip())
    outPredictLabelFile.close()

    compLabels = []
    for line in outCompareLabelFile:
        compLabels.append(line.strip())
    outCompareLabelFile.close()

    for lineNum, content in enumerate(contents):
        trueLabel = trueLabels[lineNum]
        predLabel = predLabels[lineNum]
        compLabel = compLabels[lineNum]
        if trueLabel == '1':
            if mode == 'recall':
                if predLabel == '1' and compLabel == '0':
                    print content
            elif mode == 'precision':
                if compLabel == '1' and predLabel == '0':
                    print content


if __name__ == "__main__":
    #labelStat("brandGroup", 3)
    #analysis()
    #lengthStat()
    outAnalysis(1, 'recall')
