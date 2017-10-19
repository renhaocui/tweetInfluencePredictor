__author__ = 'rencui'
import json
from os import listdir
import shutil
from os.path import isfile
from os import makedirs

def createEmptyFolders(location):
    brandList = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    for brand in brandList:
        makedirs(location+'/'+brand)


def combineFolders(fromFolder, toFolder):
    print 'combining tweet datasets'
    brandList = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    for brand in brandList:
        fromLocation = fromFolder+'/'+brand
        toLocation = toFolder+'/'+brand
        for fileName in listdir(fromLocation):
            if not isfile(toLocation+'/'+fileName):
                shutil.move(fromLocation+'/'+fileName, toLocation+'/'+fileName)


def blend(fileSize, offset):
    print 'blending tweets...'
    brandList = []
    listFile = open('brand.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    combinedOutFile = open('dataset/experiment/total.json', 'w')
    tweetCount = {}
    for brand in brandList:
        if brand not in tweetCount:
            tweetCount[brand] = 0
        tweetIDSet = set()
        tweetData = {}
        #finalTweetData = {}
        for index in range(fileSize):
            inputFile = open("../ad_data/" + brand + '/' + str(index + offset) + '.json', 'r')
            for line in inputFile:
                data = json.loads(line.strip())
                tweetID = data['id']
                if tweetID not in tweetIDSet:
                    tweetCount[brand] += 1
                    tweetIDSet.add(tweetID)
                    temp = {'id': tweetID, 'text': data['text'], 'create_at': data['created_at'], 'user_create_at': data['user']['created_at']}
                    hashtags = []
                    if 'hashtags' in data['entities']:
                        for tag in data['entities']['hashtags']:
                            hashtags.append(tag['text'])
                    temp['hashtags'] = hashtags
                    urls = []
                    if 'urls' in data['entities']:
                        for url in data['entities']['urls']:
                            urls.append(url['url'])
                    temp['urls'] = urls
                    mentions = []
                    if 'user_mentions' in data['entities']:
                        for mention in data['entities']['user_mentions']:
                            mentions.append(mention['screen_name'])
                    temp['mentions'] = mentions
                    media = []
                    if 'media' in data['entities']:
                        for item in data['entities']['media']:
                            media.append((item['media_url'], item['type']))
                    temp['media'] = media
                    temp['source'] = data['source']
                    tempList = []
                    subTemp = {'index': 0, 'favorite_count': data['favorite_count'], 'retweet_count': data['retweet_count'],
                               'user_favorite_count': data['user']['favourites_count'], 'user_followers_count': data['user']['followers_count'],
                               'user_friends_count': data['user']['friends_count'], 'user_statuses_count': data['user']['statuses_count'],
                               'user_listed_count': data['user']['listed_count']}
                    tempList.append(subTemp)
                    temp['dynamic'] = tempList
                    temp['brand'] = brand
                    tweetData[tweetID] = temp
                    #finalTweetData[tweetID] = temp
                else:
                    subTemp = {'index': len(tweetData[tweetID]['dynamic']), 'favorite_count': data['favorite_count'], 'retweet_count': data['retweet_count'],
                               'user_favorite_count': data['user']['favourites_count'], 'user_followers_count': data['user']['followers_count'],
                               'user_friends_count': data['user']['friends_count'], 'user_statuses_count': data['user']['statuses_count'],
                               'user_listed_count': data['user']['listed_count']}
                    #finalTweetData[tweetID]['dynamic'][0] = subTemp
                    tweetData[tweetID]['dynamic'].append(subTemp)
            inputFile.close()

        for (tweetID, tweet) in tweetData.items():
            combinedOutFile.write(json.dumps(tweet) + '\n')
    combinedOutFile.close()

    print tweetCount


if __name__ == "__main__":
    #blend(951, 0)
    #createEmptyFolders('/Users/rencui/Desktop/empty')
    combineFolders('/Users/rencui/Desktop/adData', '/Volumes/WD-Win/Data/ad data')