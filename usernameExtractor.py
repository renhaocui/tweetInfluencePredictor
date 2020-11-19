import time
import twitter
import json


def oauth_login():
    c_k = 'R2FZHZcAcHFatakYhKL2cQcVo'
    c_s = 'jwkcIPCkrOBdxKVTVVE7d7cIwH8ZyHHtqxYeCVUZs35Lu4BOkY'
    a_t = '141612471-3UJPl93cGf2XBm2JkBn26VFewzwK3WGN1EiKJi4T'
    a_t_s = 'do1I1vtIvjgQF3vr0ln4pYVbsAj5OZIxuuATXjgBaqUYM'

    # credentials for OAuth
    CONSUMER_KEY = c_k
    CONSUMER_SECRET = c_s
    OAUTH_TOKEN = a_t
    OAUTH_TOKEN_SECRET = a_t_s
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def extractTotalMentions():
    mentionList = set()
    inputFile = open('dataset/experiment/total.json', 'r')
    for line in inputFile:
        data = json.loads(line.strip())
        for ment in data['mentions']:
            if ment not in mentionList:
                mentionList.add(ment)
    inputFile.close()

    mentionFile = open('dataset/experiment/mention.list', 'w')
    for ment in mentionList:
        mentionFile.write(ment + '\n')
    mentionFile.close()


def collector(mentionListFilename, outputFilename):
    userFile = open(mentionListFilename, 'r')
    outputFile = open(outputFilename, 'w')

    requestLimit = 300
    twitter_api = oauth_login()
    nameList = []
    for line in userFile:
        name = line.strip()
        if name not in nameList:
            nameList.append(name)
    userFile.close()
    print(len(nameList))

    outputList = []
    tempList = []
    requestCount = 0
    print('Collecting...')
    for index, userId in enumerate(nameList):
        if requestCount > requestLimit:
            print('waiting for 15m...')
            time.sleep(900)
            requestCount = 0
        if index % 99 == 0 and index != 0:
            tempList.append(userId)
            requestCount += 1
            response = twitter_api.users.lookup(screen_name=','.join(tempList))
            tempList = []
            for user in response:
                screenName = user['screen_name']
                outputList.append(screenName)
                outputFile.write(json.dumps(user)+'\n')
        elif index == len(nameList)-1:
            tempList.append(userId)
            requestCount += 1
            response = twitter_api.users.lookup(screen_name=','.join(tempList))
            tempList = []
            for user in response:
                screenName = user['screen_name']
                outputList.append(screenName)
                outputFile.write(json.dumps(user)+'\n')
        else:
            tempList.append(userId)

    outputFile.close()
    count = 0
    for name in nameList:
        if name not in outputList:
            count += 1
            #print name
    print(count)


if __name__ == '__main__':
    #extractTotalMentions()
    collector('dataset/commTweets/mention.list', 'dataset/commTweets/mention.json')