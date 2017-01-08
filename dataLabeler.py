__author__ = 'renhao.cui'
from scipy import stats
import json
import statistics as stat
import tweetTextCleaner
import sys
from datetime import datetime
reload(sys)
sys.setdefaultencoding('utf8')

filterTerms = ['iphone 7', 'pikachu', 'pokemon go', 'macbook pro']

def label(mode):
    print 'extracting outliers...'
    brandList = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    exceptionFile = open('dataset/exceptions/exceptions.list', 'r')
    exceptionList = set()
    for line in exceptionFile:
        exceptionList.add(long(line.strip()))
    exceptionFile.close()

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
        print brand
        outLierFile = open('dataset/exceptions/'+brand+'.outliers', 'w')
        brandData = []
        brandScoreList = []

        for data in totalBrandData[brand]:
            tweetID = long(data['id'])
            if tweetID not in exceptionList:
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
                        labelScore = (2.0*retweet + favorite)*10000/followers
                        brandData.append({'brand': brand,'content': content, 'score': labelScore, 'id': tweetID, 'day': day, 'hour': hour, 'mentions': data['mentions'], 'hashtags': data['hashtags'],
                                          'author_statuses_count': author_statuses_count, 'author_favorite_count': author_favorite_count, 'author_listed_count': author_listed_count,
                                          'authorInterval': authorInterval, 'author_followers_count': followers})
                        brandScoreList.append(labelScore)

        zScores = stats.zscore(brandScoreList)
        if len(zScores) != len(brandData):
            print 'Z-score Error!'
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
        print 'mean: ' + str(meanScore)
        print 'stdev: ' + str(stdevScore)
        print 'mdean: ' + str(stat.median(normalScores))
        if stdevScore >= meanScore:
            print 'CAUTION'
        else:
            print 'PASS'
        print ''

        if mode == 1:
            # label post with 1-10 score
            cleanSize = len(cleanScore)
            binSize = cleanSize/10
            threshold = binSize
            labelScore = 1
            for count, item in enumerate(cleanData):
                if count <= threshold or labelScore == 10:
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
                        totalDisplayFile.write(brand+' | '+str(labelScore)+' | '+day+' | '+hour+' | '+unicode(item['content'], errors='ignore')+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                        item['label'] = labelScore
                        totalOutputFile.write(json.dumps(item)+'\n')
                    except:
                        print content
                else:
                    threshold += binSize
                    labelScore += 1
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

                if count <= cleanSize / 2:
                    labelScore = 1
                else:
                    labelScore = 0
                item['label'] = labelScore
                totalOutputFile.write(json.dumps(item) + '\n')
                try:
                    totalDisplayFile.write(
                        brand + ' | ' + str(labelScore) + ' | ' + day + ' | ' + hour + ' | ' + unicode(
                            item['content'], errors='ignore') + ' | ' + str(
                            item['id']) + ' | ' + hashtagOutput + ' | ' + mentionsOutput + '\n')
                except:
                    print content

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
                    totalDisplayFile.write(brand+' | '+str(score)+' | '+day+' | '+hour+' | '+unicode(item['content'], errors='ignore')+' | '+str(item['id'])+' | '+hashtagOutput+' | '+mentionsOutput+'\n')
                    item['label'] = score
                    totalOutputFile.write(json.dumps(item)+'\n')
                except:
                    print content

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


if __name__ == "__main__":
    label(2)
    #scoreFileBlender()