from scipy import stats
import json
import operator
import subprocess
import statistics as stat
import tweetTextCleaner
from sklearn.feature_extraction.text import *
from datetime import datetime
from sklearn import cluster
import numpy
#import word2vecReader
#from tokenizer import simpleTokenize

filterTerms = ['iphone 7', 'pikachu', 'pokemon go', 'macbook pro', 'trump', 'note 7']

def processDate(inputDate):
    dateTemp = inputDate.split()
    day = dateTemp[0]
    hour = dateTemp[3].split(':')[0]
    date = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
    return day, hour, datetime.strptime(date, '%b %d %Y')


def label(mode):
    tweetIDSet = set()
    print('extracting outliers...')
    brandList = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()
    '''
    exceptionFile = open('dataset/exceptions/exceptions.list', 'r')
    exceptionList = set()
    for line in exceptionFile:
        exceptionList.add(long(line.strip()))
    exceptionFile.close()
    '''
    totalDisplayFile = open('dataset/experiment/clean.display', 'w')
    totalOutputFile = open('dataset/experiment/clean.labeled', 'w')
    statFile = open('dataset/analysis/stat.total', 'w')
    #totalCleanScore = []
    #totalCleanData = []

    mentionList = set()
    hashtagList = set()
    totalBrandData = {}

    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        temp = json.loads(line.strip())
        brand = temp['brand']
        if brand not in totalBrandData:
            totalBrandData[brand] = [temp]
        else:
            totalBrandData[brand].append(temp)
    inputFile.close()

    for brand in brandList:
        print(brand)
        outLierFile = open('dataset/exceptions/'+brand+'.outliers', 'w')
        brandData = []
        brandScoreList = []

        for data in totalBrandData[brand]:
            tweetID = data['id']
            #if tweetID not in exceptionList:
            if tweetID not in tweetIDSet:
                tweetIDSet.add(tweetID)
                text = data['text'].encode('utf-8')
                filtered = False
                for term in filterTerms:
                    if term in text.lower():
                        filtered = True
                        break
                if not filtered:
                    content = tweetTextCleaner.tweetCleaner(text)
                    finalIndex = len(data['dynamic'])-1
                    retweet = float(data['dynamic'][finalIndex]['retweet_count'])
                    favorite = float(data['dynamic'][finalIndex]['favorite_count'])
                    followers = float(data['dynamic'][finalIndex]['user_followers_count'])

                    if retweet == 0:
                        ratio = 0
                    else:
                        ratio = favorite/retweet
                    statFile.write(str(favorite)+'\t'+str(retweet)+'\t'+str(followers)+'\t'+str(ratio)+'\n')

                    author_statuses_count = float(data['dynamic'][finalIndex]['user_statuses_count'])
                    author_favorite_count = float(data['dynamic'][finalIndex]['user_favorite_count'])
                    author_listed_count = float(data['dynamic'][finalIndex]['user_listed_count'])

                    dateTemp = data['create_at'].split()
                    day = dateTemp[0]
                    hour = dateTemp[3].split(':')[0]
                    postDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
                    dateTemp = data['user_create_at'].split()
                    authorDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
                    postData_object = datetime.strptime(postDate, '%b %d %Y')
                    authorData_object = datetime.strptime(authorDate, '%b %d %Y')
                    authorInterval = float((postData_object - authorData_object).days)

                    if followers > 0:
                        labelScore = (2.0 * retweet + favorite) * 10000 / followers
                        brandData.append({'brand': brand,'content': content, 'score': labelScore, 'id': tweetID, 'day': day, 'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                                          'author_statuses_count': author_statuses_count, 'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                                          'authorInterval': authorInterval, 'author_followers_count': followers})
                        brandScoreList.append(labelScore)

        zScores = stats.zscore(brandScoreList)
        if len(zScores) != len(brandData):
            print('Z-score Error!')
        outputData = []
        for index, item in enumerate(brandData):
            item['zScore'] = float(zScores[index])
            outputData.append(item)

        cleanData = []
        cleanScore = []
        sorted_output = sorted(outputData, key=lambda x: x['score'])
        for item in reversed(sorted_output):
            z = item['zScore']
            if z > 2:
                outLierFile.write(str(item['score'])+' | '+str(z)+' : '+' | '+str(item['id'])+' | '+item['content']+'\n')
            else:
                cleanData.append(item)
                cleanScore.append(item['score'])
                #totalCleanScore.append(item['score'])
                #totalCleanData.append(item)

        outLierFile.close()

        maxScore = max(cleanScore)
        minScore = min(cleanScore)
        normalScores = []
        for score in cleanScore:
            normalScores.append((score - minScore) / (maxScore - minScore))
        stdevScore = stat.stdev(normalScores)
        meanScore = stat.mean(normalScores)
        print('mean: ' + str(meanScore))
        print('stdev: ' + str(stdevScore))
        print('mdean: ' + str(stat.median(normalScores)))
        if stdevScore >= meanScore:
            print('CAUTION')
        else:
            print('PASS')
        print()

        if mode == 1:
            # label post with 1-10 score
            cleanSize = len(cleanScore)
            binSize = cleanSize/10
            threshold = binSize
            labelScore = 10
            for count, item in enumerate(cleanData):
                if count <= threshold or labelScore == 1:
                    hashtagOutput = ''
                    mentionsOutput = ''
                    for ht in item['hashtags']:
                        if ht not in hashtagList:
                            hashtagList.add(ht)
                        hashtagOutput += ht + ';'
                    if hashtagOutput == '':
                        hashtagOutput = 'NONE'
                    else:
                        hashtagOutput = hashtagOutput[:-1]
                    for ment in item['mentions']:
                        if ment not in mentionList:
                            mentionList.add(ment)
                        mentionsOutput += ment + ';'
                    if mentionsOutput == '':
                        mentionsOutput = 'NONE'
                    else:
                        mentionsOutput = mentionsOutput[:-1]
                    try:
                        totalDisplayFile.write(brand+' | '+str(labelScore)+' | '+day+' | '+hour+' | '+item['content']+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                        item['label'] = labelScore
                        totalOutputFile.write(json.dumps(item)+'\n')
                    except:
                        print(content)
                else:
                    print(threshold)
                    threshold += binSize
                    labelScore -= 1
        elif mode == 2:
            # binary label (0, 1)
            cleanSize = len(cleanScore)
            for count, item in enumerate(cleanData):
                hashtagOutput = ''
                mentionsOutput = ''
                for ht in item['hashtags']:
                    if ht not in hashtagList:
                        hashtagList.add(ht)
                    hashtagOutput += ht + ';'
                if hashtagOutput == '':
                    hashtagOutput = 'NONE'
                else:
                    hashtagOutput = hashtagOutput[:-1]
                for ment in item['mentions']:
                    if ment not in mentionList:
                        mentionList.add(ment)
                    mentionsOutput += ment + ';'
                if mentionsOutput == '':
                    mentionsOutput = 'NONE'
                else:
                    mentionsOutput = mentionsOutput[:-1]

                if count <= 0.5 * cleanSize:
                    labelScore = 1
                else:
                    labelScore = 0
                item['label'] = labelScore
                totalOutputFile.write(json.dumps(item) + '\n')
                try:
                    totalDisplayFile.write(
                        brand + ' | ' + str(labelScore) + ' | ' + day + ' | ' + hour + ' | ' + item['content'] + ' | ' + str(
                            item['id']) + ' | ' + hashtagOutput + ' | ' + mentionsOutput + '\n')
                except:
                    print(content)

        else:
            # label with normalized scores
            scoreDistFile = open('dataset/stats/scoreDist.'+brand, 'w')
            for index, normalScore in enumerate(normalScores):
                item = cleanData[index]
                score = normalScore * 10
                scoreDistFile.write(str(score)+'\n')
                hashtagOutput = ''
                mentionsOutput = ''
                for ht in item['hashtags']:
                    if ht not in hashtagList:
                        hashtagList.add(ht)
                    hashtagOutput += ht + ';'
                if hashtagOutput == '':
                    hashtagOutput = 'NONE'
                else:
                    hashtagOutput = hashtagOutput[:-1]
                for ment in item['mentions']:
                    if ment not in mentionList:
                        mentionList.add(ment)
                    mentionsOutput += ment + ';'
                if mentionsOutput == '':
                    mentionsOutput = 'NONE'
                else:
                    mentionsOutput = mentionsOutput[:-1]
                try:
                    totalDisplayFile.write(brand+' | '+str(score)+' | '+day+' | '+hour+' | '+item['content']+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                    item['label'] = score
                    totalOutputFile.write(json.dumps(item)+'\n')
                except:
                    print(content)

            scoreDistFile.close()

    hashtagFile = open('dataset/experiment/hashtag.list', 'w')
    mentionFile = open('dataset/experiment/mention.list', 'w')
    for ht in hashtagList:
        hashtagFile.write(ht+'\n')
    for ment in mentionList:
        mentionFile.write(ment+'\n')

    hashtagFile.close()
    mentionFile.close()
    statFile.close()
    totalOutputFile.close()


def label_new(mode, inputFile):
    totalDisplayFile = open('dataset/commTweets/clean.display', 'w')
    totalOutputFile = open('dataset/commTweets/clean.json', 'w')

    mentionList = set()
    hashtagList = set()
    totalBrandData = {}

    inputFile = open(inputFile, 'r')
    for line in inputFile:
        temp = json.loads(line.strip())
        brand = temp['brand']
        if brand not in totalBrandData:
            totalBrandData[brand] = [temp]
        else:
            totalBrandData[brand].append(temp)
    inputFile.close()

    for brand in totalBrandData:
        print(brand)
        outLierFile = open('dataset/commTweets/outliers/'+brand+'.outliers', 'w')
        brandData = []
        brandScoreList = []

        for data in totalBrandData[brand]:
            tweetID = data['id']
            text = data['text']
            content = tweetTextCleaner.tweetCleaner(text)
            retweet = float(data['retweet_count'])
            favorite = float(data['favorite_count'])
            followers = float(data['user_followers_count'])
            author_statuses_count = float(data['user_statuses_count'])
            author_favorite_count = float(data['user_favorite_count'])
            author_listed_count = float(data['user_listed_count'])

            day, hour, postData_object = processDate(data['create_at'])
            _, _, authorData_object = processDate(data['user_create_at'])
            authorInterval = float((postData_object - authorData_object).days)

            if followers > 0:
                labelScore = (2.0 * retweet + favorite) * 10000 / followers
                brandData.append({'brand': brand, 'content': content, 'score': labelScore, 'id': tweetID, 'day': day, 'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                                  'author_statuses_count': author_statuses_count, 'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                                  'authorInterval': authorInterval, 'author_followers_count': followers})
                brandScoreList.append(labelScore)

        zScores = stats.zscore(brandScoreList)
        if len(zScores) != len(brandData):
            print('Z-score Error!')
        outputData = []
        for index, item in enumerate(brandData):
            item['zScore'] = float(zScores[index])
            outputData.append(item)

        cleanData = []
        cleanScore = []
        sorted_output = sorted(outputData, key=lambda x: x['score'])
        for item in reversed(sorted_output):
            z = item['zScore']
            if z > 2:
                outLierFile.write(str(item['score'])+' | '+str(z)+' : '+' | '+str(item['id'])+' | '+item['content']+'\n')
            else:
                cleanData.append(item)
                cleanScore.append(item['score'])
                #totalCleanScore.append(item['score'])
                #totalCleanData.append(item)

        outLierFile.close()

        maxScore = max(cleanScore)
        minScore = min(cleanScore)
        normalScores = []
        for score in cleanScore:
            normalScores.append((score - minScore) / (maxScore - minScore))
        stdevScore = stat.stdev(normalScores)
        meanScore = stat.mean(normalScores)
        #print('mean: ' + str(meanScore))
        #print('stdev: ' + str(stdevScore))
        #print('mdean: ' + str(stat.median(normalScores)))
        if stdevScore >= meanScore:
            print('CAUTION')
        else:
            print('PASS')
        print()

        if mode == 1:
            # label post with 1-10 score
            cleanSize = len(cleanScore)
            binSize = cleanSize/10
            threshold = binSize
            labelScore = 10
            for count, item in enumerate(cleanData):
                if count <= threshold or labelScore == 1:
                    hashtagOutput = ''
                    mentionsOutput = ''
                    for ht in item['hashtags']:
                        if ht not in hashtagList:
                            hashtagList.add(ht)
                        hashtagOutput += ht + ';'
                    hashtagOutput = 'NONE' if hashtagOutput == '' else hashtagOutput[:-1]
                    for ment in item['mentions']:
                        if ment not in mentionList:
                            mentionList.add(ment)
                        mentionsOutput += ment + ';'
                    mentionsOutput = 'NONE' if mentionsOutput == '' else mentionsOutput[:-1]
                    try:
                        totalDisplayFile.write(brand+' | '+str(labelScore)+' | '+day+' | '+hour+' | '+item['content']+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                        item['label'] = labelScore
                        totalOutputFile.write(json.dumps(item)+'\n')
                    except:
                        print(content)
                else:
                    #print(threshold)
                    threshold += binSize
                    labelScore -= 1
        elif mode == 2:
            # binary label (0, 1)
            cleanSize = len(cleanScore)
            for count, item in enumerate(cleanData):
                hashtagOutput = ''
                mentionsOutput = ''
                for ht in item['hashtags']:
                    if ht not in hashtagList:
                        hashtagList.add(ht)
                    hashtagOutput += ht + ';'
                if hashtagOutput == '':
                    hashtagOutput = 'NONE'
                else:
                    hashtagOutput = hashtagOutput[:-1]
                for ment in item['mentions']:
                    if ment not in mentionList:
                        mentionList.add(ment)
                    mentionsOutput += ment + ';'
                if mentionsOutput == '':
                    mentionsOutput = 'NONE'
                else:
                    mentionsOutput = mentionsOutput[:-1]

                if count <= 0.5 * cleanSize:
                    labelScore = 1
                else:
                    labelScore = 0
                item['label'] = labelScore
                totalOutputFile.write(json.dumps(item) + '\n')
                try:
                    totalDisplayFile.write(
                        brand + ' | ' + str(labelScore) + ' | ' + day + ' | ' + hour + ' | ' + item['content'] + ' | ' + str(
                            item['id']) + ' | ' + hashtagOutput + ' | ' + mentionsOutput + '\n')
                except:
                    print(content)

        else:
            # label with normalized scores
            scoreDistFile = open('dataset/stats/scoreDist.'+brand, 'w')
            for index, normalScore in enumerate(normalScores):
                item = cleanData[index]
                score = normalScore * 10
                scoreDistFile.write(str(score)+'\n')
                hashtagOutput = ''
                mentionsOutput = ''
                for ht in item['hashtags']:
                    if ht not in hashtagList:
                        hashtagList.add(ht)
                    hashtagOutput += ht + ';'
                if hashtagOutput == '':
                    hashtagOutput = 'NONE'
                else:
                    hashtagOutput = hashtagOutput[:-1]
                for ment in item['mentions']:
                    if ment not in mentionList:
                        mentionList.add(ment)
                    mentionsOutput += ment + ';'
                if mentionsOutput == '':
                    mentionsOutput = 'NONE'
                else:
                    mentionsOutput = mentionsOutput[:-1]
                try:
                    totalDisplayFile.write(brand+' | '+str(score)+' | '+day+' | '+hour+' | '+item['content']+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                    item['label'] = score
                    totalOutputFile.write(json.dumps(item)+'\n')
                except:
                    print(content)

            scoreDistFile.close()

    hashtagFile = open('dataset/commTweets/hashtag.list', 'w')
    mentionFile = open('dataset/commTweets/mention.list', 'w')
    for ht in hashtagList:
        hashtagFile.write(ht+'\n')
    for ment in mentionList:
        mentionFile.write(ment+'\n')

    hashtagFile.close()
    mentionFile.close()
    totalOutputFile.close()


def groupSampler(groupMode, groupSize, seed):
    print(groupMode)
    inputFile = open('dataset/experiment/labeled_data/' + groupMode + '_' + str(groupSize) + '.labeled', 'r')
    groupData = {}
    for num in range(int(groupSize)):
        groupData[num] = {}

    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['content'].encode('utf-8').replace('\n', ' ').replace('\r', ' ')
        group = data['group']
        groupData[group][tweetID] = text
    inputFile.close()

    outputFile = open('dataset/experiment/sample/' + groupMode + '_' + str(groupSize) + '.sample', 'w')
    for groupIndex in range(int(groupSize)):
        outputFile.write('Group: ' + str(groupIndex)+'\n')
        print(len(groupData[groupIndex]))
        for count, tweetID in enumerate(groupData[groupIndex]):
            if count % seed == 0:
                outputFile.write(groupData[groupIndex][tweetID]+'\t'+str(tweetID)+'\n')
    outputFile.close()


def brandLabel(removeOutliers=True):
    if removeOutliers:
        totalOutputFile = open('dataset/experiment/brandGroup_0.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/brandGroup_0.content', 'w')
        statFile = open('dataset/analysis/brandGroup_0.stat', 'w')
    else:
        totalOutputFile = open('dataset/experiment/brandGroup_0__full' + '.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/brandGroup_0__full' + '.content', 'w')
        statFile = open('dataset/analysis/brandGroup_0_full' + '.stat', 'w')
    totalData = {}

    brandGroupData = {}
    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        filtered = False
        for term in filterTerms:
            if term in text.lower():
                filtered = True
                break
        if not filtered:
            brand = data['brand']
            if brand not in brandGroupData:
                brandGroupData[brand] = []
            brandGroupData[brand].append(tweetID)
            content = tweetTextCleaner.tweetCleaner(text)
            finalIndex = len(data['dynamic']) - 1
            retweet = float(data['dynamic'][finalIndex]['retweet_count'])
            favorite = float(data['dynamic'][finalIndex]['favorite_count'])
            followers = float(data['dynamic'][finalIndex]['user_followers_count'])
            if retweet == 0:
                ratio = 0
            else:
                ratio = favorite / retweet
            statFile.write(
                str(favorite) + '\t' + str(retweet) + '\t' + str(followers) + '\t' + str(ratio) + '\n')
            author_statuses_count = float(data['dynamic'][finalIndex]['user_statuses_count'])
            author_favorite_count = float(data['dynamic'][finalIndex]['user_favorite_count'])
            author_listed_count = float(data['dynamic'][finalIndex]['user_listed_count'])
            dateTemp = data['create_at'].split()
            day = dateTemp[0]
            hour = dateTemp[3].split(':')[0]
            postDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            dateTemp = data['user_create_at'].split()
            authorDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            postData_object = datetime.strptime(postDate, '%b %d %Y')
            authorData_object = datetime.strptime(authorDate, '%b %d %Y')
            authorInterval = float((postData_object - authorData_object).days)
            if followers > 0:
                successScore = (2.0 * retweet + favorite) * 10000 / followers
                temp = {'brand': brand, 'content': content, 'success_score': successScore, 'id': tweetID, 'day': day,
                             'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                             'author_statuses_count': author_statuses_count,
                             'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                             'authorInterval': authorInterval, 'author_followers_count': followers}
                totalData[tweetID] = temp
    inputFile.close()

    for brand, tweetIDs in brandGroupData.items():
        print('Brand: ' + brand)
        groupScoreList = []
        IDList = []
        for tweetID in tweetIDs:
            if tweetID in totalData:
                successScore = totalData[tweetID]['success_score']
                groupScoreList.append(successScore)
                IDList.append(tweetID)

        cleanDataList = []
        if removeOutliers:
            zScores = stats.zscore(groupScoreList)
            if len(zScores) != len(groupScoreList):
                print ('Z-score Error!')
        for index, item in enumerate(IDList):
            if removeOutliers:
                zScore = float(zScores[index])
                if zScore <= 2:
                    cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
            else:
                cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
        print('Group Size: ' + str(len(cleanDataList)))
        sorted_cleanDataList = sorted(cleanDataList, key=lambda x: x['success_score'], reverse=True)

        # label post with 1-10 score
        cleanSize = len(cleanDataList)
        binSize = cleanSize / 10
        threshold = binSize
        labelScore = 10
        for count, item in enumerate(sorted_cleanDataList):
            tweetID = item['id']
            if count <= threshold or labelScore == 1:
                tempData = totalData[tweetID]
                tempData['label'] = labelScore
                tempData['group'] = brand
                totalOutputFile.write(json.dumps(tempData) + '\n')
                contentOutputFile.write(tempData['content']+'\n')
            else:
                #print threshold
                threshold += binSize
                labelScore -= 1

    statFile.close()
    totalOutputFile.close()
    contentOutputFile.close()


def groupLabel(groupMode, groupSize, removeOutliers=True):
    groupFile = open('dataset/experiment/group_indicies/'+groupMode+'.'+str(groupSize), 'r')
    for line in groupFile:
        groupData = json.loads(line.strip())
    groupFile.close()

    if removeOutliers:
        totalOutputFile = open('dataset/experiment/labeled_data/'+groupMode+'_'+str(groupSize)+'.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/'+groupMode+'_'+str(groupSize)+'.content', 'w')
        statFile = open('dataset/analysis/'+groupMode+'_'+str(groupSize)+'.stat', 'w')
    else:
        totalOutputFile = open('dataset/experiment/labeled_data/' + groupMode + '_' + str(groupSize) + '_full' + '.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/' + groupMode + '_' + str(groupSize) + '_full' + '.content', 'w')
        statFile = open('dataset/analysis/' + groupMode + '_' + str(groupSize) + '_full' + '.stat', 'w')
    totalData = {}

    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        filtered = False
        for term in filterTerms:
            if term in text.lower():
                filtered = True
                break
        if not filtered:
            brand = data['brand']
            content = tweetTextCleaner.tweetCleaner(text)
            finalIndex = len(data['dynamic']) - 1
            retweet = float(data['dynamic'][finalIndex]['retweet_count'])
            favorite = float(data['dynamic'][finalIndex]['favorite_count'])
            followers = float(data['dynamic'][finalIndex]['user_followers_count'])
            if retweet == 0:
                ratio = 0
            else:
                ratio = favorite / retweet
            statFile.write(
                str(favorite) + '\t' + str(retweet) + '\t' + str(followers) + '\t' + str(ratio) + '\n')
            author_statuses_count = float(data['dynamic'][finalIndex]['user_statuses_count'])
            author_favorite_count = float(data['dynamic'][finalIndex]['user_favorite_count'])
            author_listed_count = float(data['dynamic'][finalIndex]['user_listed_count'])
            dateTemp = data['create_at'].split()
            day = dateTemp[0]
            hour = dateTemp[3].split(':')[0]
            postDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            dateTemp = data['user_create_at'].split()
            authorDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            postData_object = datetime.strptime(postDate, '%b %d %Y')
            authorData_object = datetime.strptime(authorDate, '%b %d %Y')
            authorInterval = float((postData_object - authorData_object).days)
            if followers > 0:
                successScore = (2.0 * retweet + favorite) * 10000 / followers
                temp = {'brand': brand, 'content': content, 'success_score': successScore, 'id': tweetID, 'day': day,
                             'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                             'author_statuses_count': author_statuses_count,
                             'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                             'authorInterval': authorInterval, 'author_followers_count': followers}
                totalData[tweetID] = temp
    inputFile.close()

    for groupIndex in range(int(groupSize)):
        print(groupMode+': ' + str(groupIndex))
        groupScoreList = []
        IDList = []
        for tweetID in groupData[str(groupIndex)]:
            if tweetID in totalData:
                successScore = totalData[tweetID]['success_score']
                groupScoreList.append(successScore)
                IDList.append(tweetID)

        cleanDataList = []
        if removeOutliers:
            zScores = stats.zscore(groupScoreList)
            if len(zScores) != len(groupScoreList):
                print('Z-score Error!')
        for index, item in enumerate(IDList):
            if removeOutliers:
                zScore = float(zScores[index])
                if zScore <= 2:
                    cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
            else:
                cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
        print('Group Size: ' + str(len(cleanDataList)))
        sorted_cleanDataList = sorted(cleanDataList, key=lambda x: x['success_score'], reverse=True)

        # label post with 1-10 score
        cleanSize = len(cleanDataList)
        binSize = cleanSize / 10
        threshold = binSize
        labelScore = 10
        for count, item in enumerate(sorted_cleanDataList):
            tweetID = item['id']
            if count <= threshold or labelScore == 1:
                tempData = totalData[tweetID]
                tempData['label'] = labelScore
                tempData['group'] = groupIndex
                totalOutputFile.write(json.dumps(tempData) + '\n')
                contentOutputFile.write(tempData['content']+'\n')
            else:
                #print threshold
                threshold += binSize
                labelScore -= 1

    statFile.close()
    totalOutputFile.close()
    contentOutputFile.close()


def simpleLabel(groupVersion, removeOutliers=True):
    if removeOutliers:
        totalOutputFile = open('dataset/experiment/labeled_data/simple_'+str(groupVersion)+'.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/simple_'+str(groupVersion)+'.content', 'w')
        statFile = open('dataset/analysis/simple_'+str(groupVersion)+'.stat', 'w')
    else:
        totalOutputFile = open('dataset/experiment/labeled_data/simple_'+str(groupVersion)+'_full.labeled', 'w')
        contentOutputFile = open('dataset/experiment/content/simple_'+str(groupVersion)+'_full.content', 'w')
        statFile = open('dataset/analysis/simple_'+str(groupVersion)+'_full.stat', 'w')
    totalData = {}

    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        filtered = False
        for term in filterTerms:
            if term in text.lower():
                filtered = True
                break
        if not filtered:
            brand = data['brand']
            content = tweetTextCleaner.tweetCleaner(text)
            finalIndex = len(data['dynamic']) - 1
            retweet = float(data['dynamic'][finalIndex]['retweet_count'])
            favorite = float(data['dynamic'][finalIndex]['favorite_count'])
            followers = float(data['dynamic'][finalIndex]['user_followers_count'])
            if retweet == 0:
                ratio = 0
            else:
                ratio = favorite / retweet
            statFile.write(
                str(favorite) + '\t' + str(retweet) + '\t' + str(followers) + '\t' + str(ratio) + '\n')
            author_statuses_count = float(data['dynamic'][finalIndex]['user_statuses_count'])
            author_favorite_count = float(data['dynamic'][finalIndex]['user_favorite_count'])
            author_listed_count = float(data['dynamic'][finalIndex]['user_listed_count'])
            dateTemp = data['create_at'].split()
            day = dateTemp[0]
            hour = dateTemp[3].split(':')[0]
            postDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            dateTemp = data['user_create_at'].split()
            authorDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            postData_object = datetime.strptime(postDate, '%b %d %Y')
            authorData_object = datetime.strptime(authorDate, '%b %d %Y')
            authorInterval = float((postData_object - authorData_object).days)
            if followers > 0:
                successScore = (2.0 * retweet + favorite) * 10000 / followers
                temp = {'brand': brand, 'content': content, 'success_score': successScore, 'id': tweetID, 'day': day,
                             'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                             'author_statuses_count': author_statuses_count,
                             'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                             'authorInterval': authorInterval, 'author_followers_count': followers}
                totalData[tweetID] = temp
    inputFile.close()

    groupScoreList = []
    IDList = []
    for tweetID in totalData:
        successScore = totalData[tweetID]['success_score']
        groupScoreList.append(successScore)
        IDList.append(tweetID)

    cleanDataList = []
    if removeOutliers:
        zScores = stats.zscore(groupScoreList)
        if len(zScores) != len(groupScoreList):
            print('Z-score Error!')
    for index, item in enumerate(IDList):
        if removeOutliers:
            zScore = float(zScores[index])
            if zScore <= 2:
                cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
        else:
            cleanDataList.append({'id': item, 'success_score': groupScoreList[index]})
    print('Group Size: ' + str(len(cleanDataList)))
    sorted_cleanDataList = sorted(cleanDataList, key=lambda x: x['success_score'], reverse=True)

    # label post with 1-10 score
    cleanSize = len(cleanDataList)
    binSize = cleanSize / 10
    threshold = binSize
    labelScore = 10
    for count, item in enumerate(sorted_cleanDataList):
        tweetID = item['id']
        if count <= threshold or labelScore == 1:
            tempData = totalData[tweetID]
            tempData['label'] = labelScore
            tempData['group'] = 0
            totalOutputFile.write(json.dumps(tempData) + '\n')
            contentOutputFile.write(tempData['content']+'\n')
        else:
            #print threshold
            threshold += binSize
            labelScore -= 1

    statFile.close()
    totalOutputFile.close()
    contentOutputFile.close()


def keywordLabel(keyword):
    outputFile = open('dataset/experiment/'+keyword+'.labeled', 'w')
    statFile = open('dataset/analysis/'+keyword+'.stat', 'w')

    tweetData = {}
    dataList = []
    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        if keyword in text.lower():
            brand = data['brand']
            content = tweetTextCleaner.tweetCleaner(text)
            finalIndex = len(data['dynamic']) - 1
            retweet = float(data['dynamic'][finalIndex]['retweet_count'])
            favorite = float(data['dynamic'][finalIndex]['favorite_count'])
            followers = float(data['dynamic'][finalIndex]['user_followers_count'])
            if retweet == 0:
                ratio = 0
            else:
                ratio = favorite / retweet
            statFile.write(
                str(favorite) + '\t' + str(retweet) + '\t' + str(followers) + '\t' + str(ratio) + '\n')
            author_statuses_count = float(data['dynamic'][finalIndex]['user_statuses_count'])
            author_favorite_count = float(data['dynamic'][finalIndex]['user_favorite_count'])
            author_listed_count = float(data['dynamic'][finalIndex]['user_listed_count'])
            dateTemp = data['create_at'].split()
            day = dateTemp[0]
            hour = dateTemp[3].split(':')[0]
            postDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            dateTemp = data['user_create_at'].split()
            authorDate = dateTemp[1] + ' ' + dateTemp[2] + ' ' + dateTemp[5]
            postData_object = datetime.strptime(postDate, '%b %d %Y')
            authorData_object = datetime.strptime(authorDate, '%b %d %Y')
            authorInterval = float((postData_object - authorData_object).days)
            if followers > 0:
                successScore = (2.0 * retweet + favorite) * 10000 / followers
                temp = {'brand': brand, 'content': content, 'success_score': successScore, 'id': tweetID, 'day': day,
                             'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                             'author_statuses_count': author_statuses_count,
                             'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                             'authorInterval': authorInterval, 'author_followers_count': followers}
                tweetData[tweetID] = temp
                dataList.append({'id': tweetID, 'success_score': successScore})
    inputFile.close()
    print(len(dataList))
    sorted_dataList = sorted(dataList, key=lambda x: x['success_score'], reverse=True)

    # label post with 1-10 score
    dataSize = len(dataList)
    binSize = dataSize / 10
    threshold = binSize
    labelScore = 10
    for count, item in enumerate(sorted_dataList):
        tweetID = item['id']
        if count <= threshold or labelScore == 1:
            tempData = tweetData[tweetID]
            tempData['label'] = labelScore
            tempData['keyword'] = keyword
            outputFile.write(json.dumps(tempData) + '\n')
        else:
            threshold += binSize
            labelScore -= 1

    statFile.close()
    outputFile.close()


def scoreFileBlender():
    data = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brand = line.strip()
        inputFile = open('dataset/stats/scoreDist.' + brand, 'r')
        for line in inputFile:
            data.append(float(line.strip()))
        inputFile.close()
    listFile.close()

    sorted_data = sorted(data, reverse=True)

    outputFile = open('dataset/stats/scoreDist.total', 'w')
    for num in sorted_data:
        outputFile.write(str(num)+'\n')
    outputFile.close()


def maxIndex(input, num):
    line = {}
    for index in range(len(input)):
        line[index] = float(input[index])
    sorted_line = sorted(line.iteritems(), key=operator.itemgetter(1), reverse=True)
    output = []
    for i in range(num):
        output.append(sorted_line[i][0])
    return output


def dataGrouper(groupMode, groupSize, hierarchical=False):
    inputFile = open('dataset/experiment/total.json', 'r')
    tweetData = []
    outputData = {}
    for index in range(int(groupSize)):
        outputData[str(index)] = []

    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        content = text.replace('\r', ' ').replace('\n', ' ')
        brand = data['brand']
        tweetData.append({'id': tweetID, 'content': content, 'brand': brand})
    inputFile.close()

    if groupMode == 'brandGroup':
        print('running brand grouping...')
        brandMapper = {}
        groupFile = open('brandGroup.list', 'r')
        for index, line in enumerate(groupFile):
            brands = line.strip().split()
            for brand in brands:
                brandMapper[brand] = str(index)
        groupFile.close()

        for tweet in tweetData:
            if tweet['brand'] in brandMapper:
                outputData[brandMapper[tweet['brand']]].append(tweet['id'])
    elif groupMode == 'topicGroup':
        print('running LDA grouping...')
        csvFile = open('TMT/LDAinput.csv', 'w')
        for tweet in tweetData:
            csvFile.write(tweetTextCleaner.tweetCleaner(tweet['content']).replace('"', '\'') + '\n')
        csvFile.close()

        subprocess.check_output('java -Xmx1024m -jar TMT/tmt-0.4.0.jar TMT/assign.scala', shell=True)

        distFile = open('TMTSnapshots/document-topic-distributions.csv', 'r')
        topicOut = {}
        for line in distFile:
            seg = line.strip().split(',')
            if seg[1] != 'NaN':
                topicOutList = maxIndex(seg[1:], int(groupSize))
                topicOut[int(seg[0])] = topicOutList
        distFile.close()

        for index, value in topicOut.items():
            outputData[str(value[0])].append(tweetData[index]['id'])
    elif groupMode == 'simGroup_binary':
        print('running kmeans clustering with binary representation...')
        tweets = []
        for tweet in tweetData:
            tweets.append(tweetTextCleaner.tweetCleaner(tweet['content']))

        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', binary='True')
        matrix = vectorizer.fit_transform(tweets)
        print(matrix.shape)
        if hierarchical:
            print()
            #z = cluster.hierarchy.linkage(matrix, 'ward')
        else:
            kmeans = cluster.KMeans(n_clusters=int(groupSize), init='k-means++')
            kmeans.fit(matrix)
            for index, label in enumerate(kmeans.labels_):
                outputData[str(label)].append(tweetData[index]['id'])

    elif groupMode == 'simGroup_emb':
        print('running kmeans clustering with CMU encoding...')
        '''
        contentFile = open('embedding/CMU_hashtag/tweet.content', 'w')
        for tweet in tweetData:
            contentFile.write(tweet['content']+'\n')
        contentFile.close()

        subprocess.check_output('python embedding/CMU_hashtag/preprocess.py', shell=True)
        subprocess.check_output('python embedding/CMU_hashtag/encode_char.py embedding/CMU_hashtag/tweet.input embedding/CMU_hashtag/best_model embedding/CMU_hashtag/', shell=True)
        '''
        embData = numpy.load('embedding/CMU_hashtag/embeddings.npy')
        print(len(embData))
        if hierarchical:
            print()
        else:
            kmeans = cluster.KMeans(n_clusters=int(groupSize), init='k-means++')
            kmeans.fit(embData)
            for index, label in enumerate(kmeans.labels_):
                outputData[str(label)].append(tweetData[index]['id'])

    outputFile = open('dataset/experiment/group_indicies/'+groupMode + '.' + str(groupSize), 'w')
    outputFile.write(json.dumps(outputData))
    outputFile.close()


'''
def content2vec(model, content):
    words = simpleTokenize(content)
    tempList = []
    for word in words:
        if word in model.vocab:
            tempList.append(model[word])
    if len(tempList) < 1:
        return numpy.zeros(400)
    vecSize = len(tempList[0])
    sumList = []
    for i in range(vecSize):
        sumList.append(0.0)
    for vec in tempList:
        for i in range(vecSize):
            sumList[i] += vec[i]
    output = []
    dataSize = len(tempList)
    for value in sumList:
        output.append(value/dataSize)
    return numpy.array(output)
'''


'''
def dataGrouperKey(groupMode, groupSize):
    keyData = {}
    keyFile = open('dataset/experiment/parser/total.key', 'r')
    for line in keyFile:
        if line.strip().startswith(':: '):
            keyData[int(line.strip().replace(':: ', ''))] = 'NONE'
        else:
            temp = line.strip().split(' :: ')
            keyData[int(temp[1])] = temp[0]
    keyFile.close()

    inputFile = open('dataset/experiment/total.json', 'r')
    tweetData = []
    outputData = {}
    for index in range(int(groupSize)):
        outputData[str(index)] = []

    for line in inputFile:
        data = json.loads(line.strip())
        tweetID = data['id']
        text = data['text'].encode('utf-8')
        key = keyData[tweetID]
        content = text.replace('\r', ' ').replace('\n', ' ')
        brand = data['brand']
        tweetData.append({'id': tweetID, 'content': content, 'brand': brand, 'key': key})
    inputFile.close()

    if groupMode == 'topicGroup':
        print('running LDA grouping...')
        csvFile = open('TMT/LDAinput.csv', 'w')
        for tweet in tweetData:
            csvFile.write(tweet['key'].replace('"', '\'') + '\n')
        csvFile.close()

        subprocess.check_output('java -Xmx1024m -jar TMT/tmt-0.4.0.jar TMT/assign.scala', shell=True)

        distFile = open('TMTSnapshots/document-topic-distributions.csv', 'r')
        topicOut = {}
        for line in distFile:
            seg = line.strip().split(',')
            if seg[1] != 'NaN':
                topicOutList = maxIndex(seg[1:], int(groupSize))
                topicOut[int(seg[0])] = topicOutList
        distFile.close()

        for index, value in topicOut.items():
            outputData[str(value[0])].append(tweetData[index]['id'])
    elif groupMode == 'simGroup_binary':
        print('running kmeans clustering with binary representation...')
        tweets = []
        for tweet in tweetData:
            tweets.append(tweet['key'])

        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', binary='True')
        matrix = vectorizer.fit_transform(tweets)
        print(matrix.shape)
        kmeans = cluster.KMeans(n_clusters=int(groupSize), init='k-means++')
        kmeans.fit(matrix)
        for index, label in enumerate(kmeans.labels_):
            outputData[str(label)].append(tweetData[index]['id'])
    elif groupMode == 'simGroup_emb':
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        contents = []
        for tweet in tweetData:
            tweetVec = content2vec(embModel, tweet['key'])
            contents.append(tweetVec)
        matrix = numpy.array(contents)
        print(matrix.shape)
        kmeans = cluster.KMeans(n_clusters=int(groupSize), init='k-means++')
        kmeans.fit(matrix)
        for index, label in enumerate(kmeans.labels_):
            outputData[str(label)].append(tweetData[index]['id'])


    outputFile = open('dataset/experiment/group_indicies/' + groupMode + '.' + str(groupSize), 'w')
    outputFile.write(json.dumps(outputData))
    outputFile.close()
'''

def dataAligner(groupMode, groupSize):
    tweetData = {}
    inputDataFile = open('dataset/experiment/'+groupMode+'_'+str(groupSize)+'.labeled', 'r')
    for line in inputDataFile:
        temp = json.loads(line.strip())
        tweetData[str(temp['id'])] = temp['label']
    orderTweetIDList = []
    cleanDataFile = open('dataset/experiment/clean.labeled', 'r')
    for line in cleanDataFile:
        temp = json.loads(line.strip())
        orderTweetIDList.append(temp['id'])



if __name__ == "__main__":
    label_new(1, 'dataset/commTweets.json')
    #label2(1)
    #scoreFileBlender()

    #dataGrouper('topicGroup', 7.2)
    #dataGrouperKey('topicGroup', 2.4)

    #groupLabel('topicGroup', 2.4, True)
    #simpleLabel(1.1, True)

    #groupSampler('simGroup_emb', 5.4, 300)
    #groupSampler('topicGroup', 2.2, 3000)
    #groupSampler('topicGroup', 2.1, 1000)
    #groupSampler('topicGroup', 2.2, 1000)
    #brandLabel()
    #keywordLabel('trump')
    #keywordLabel('iphone')