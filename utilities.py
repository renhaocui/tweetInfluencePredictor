import re
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sklearn.metrics
import json
import pickle
import gensim
import word2vecReader
from afinn import Afinn
from nltk import word_tokenize
from textstat.textstat import textstat

MAX_LEN = 40
MAX_NB_WORDS = 10000
EMB_SIZE = 300
EMB_PATH = 'crawl-300d-2M.vec'

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
posList = ['N', 'V', 'A']

def POSRatio(inputDict):
    out = []
    temp = []
    for pos in posList:
        temp.append(float(inputDict[pos]))
    if sum(temp) == 0:
        out = [0.0, 0.0, 0.0]
    else:
        for item in temp:
            out.append(item/sum(temp))
    return out


def hourMapper(hour):
    input = int(hour)
    if 0 <= input < 6:
        output = 0
    elif 6 <= input < 12:
        output = 1
    elif 12 <= input < 18:
        output = 2
    else:
        output = 3
    return output


def mapMention(inputFile):
    mentionFile = open(inputFile, 'r')
    outputMapper = {}
    for line in mentionFile:
        mention = json.loads(line.strip())
        if mention['verified'] == 'true':
            verify = 1
        else:
            verify = 0
        outputMapper[mention['screen_name']] = (verify, mention['followers_count'])
    mentionFile.close()
    return outputMapper


def vectorizeWord(content, corpus):
    vector = {}
    output = []
    for word in content:
        if word not in vector:
            vector[word] = 1.0
        else:
            vector[word] += 1.0
    for word in corpus:
        output.append(vector[word])
    return output


def readSentimentList(file_name):
    ifile = open(file_name, 'r')
    happy_log_probs = {}
    sad_log_probs = {}
    ifile.readline() #Ignore title row

    for line in ifile:
        tokens = line[:-1].split(',')
        happy_log_probs[tokens[0]] = float(tokens[1])
        sad_log_probs[tokens[0]] = float(tokens[2])

    return happy_log_probs, sad_log_probs

def classifySentiment(words, happy_log_probs, sad_log_probs):
    # Get the log-probability of each word under each sentiment
    happy_probs = [happy_log_probs[word] for word in words if word in happy_log_probs]
    sad_probs = [sad_log_probs[word] for word in words if word in sad_log_probs]

    # Sum all the log-probabilities for each sentiment to get a log-probability for the whole tweet
    tweet_happy_log_prob = numpy.sum(happy_probs)
    tweet_sad_log_prob = numpy.sum(sad_probs)

    # Calculate the probability of the tweet belonging to each sentiment
    prob_happy = numpy.reciprocal(numpy.exp(tweet_sad_log_prob - tweet_happy_log_prob) + 1)
    prob_sad = 1 - prob_happy

    return prob_happy, prob_sad

def content2vec(model, content):
    words = word_tokenize(content)
    tempList = []
    for word in words:
        if word in model.vocab:
            tempList.append(model[word])
    if len(tempList) < 1:
        return None
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
        print('prediction error!')
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
    else:
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


def evaluate2(predictions, test_labels):
    if len(predictions) != len(test_labels):
        print('prediction error!')
        return 404
    precision = sklearn.metrics.precision_score(test_labels, predictions)
    recall = sklearn.metrics.recall_score(test_labels, predictions)
    F1 = sklearn.metrics.f1_score(test_labels, predictions)
    auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
    return precision, recall, F1, auc



def shrinkPuncuation(input):
    input = re.sub('\.+', '.', input)
    input = re.sub(',+', ',', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('=+', '=', input)
    input = re.sub('-+', '-', input)
    input = re.sub('_+', '_', input)
    return input

def tokenizeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, '<URL>')
    return input


def getEmbeddings(embeddingMode, contents, ids):
    if embeddingMode == 1:
        embeddingFeature = numpy.load('dataset/experiment/vector/CMU_total_emd.npy')
    elif embeddingMode == 2:
        model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/general_tweet.d2v')
        tempFeatures = []
        for content in contents:
            words = word_tokenize(content)
            tempFeatures.append(numpy.array(model.infer_vector(words)))
        embeddingFeature = numpy.array(tempFeatures)
    elif embeddingMode == 3:
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tempFeatures = []
        for id, content in zip(ids, contents):
            tweetVec = content2vec(embModel, content)
            if tweetVec is None:
                tempFeatures.append(numpy.zeros(400))
            else:
                tempFeatures.append(tweetVec)
        embeddingFeature = numpy.array(tempFeatures)

    return embeddingFeature


def loadData(dataFilename):
    print('Loading Data...')
    ids = []
    contents = []
    scores = []
    days = []
    time = []
    labels = []
    usernames = []
    authorFollowers = []
    authorStatusCount = []
    authorFavoriteCount = []
    authorListedCount = []
    authorIntervals = []
    parseLength = []
    headCount = []
    POScounts = []

    inputFile = open(dataFilename, 'r')
    for index, line in enumerate(inputFile):
        item = json.loads(line.strip())
        ids.append(str(item['id']))
        contents.append(item['content'])
        labels.append(item['label'])
        time.append(hourMapper(item['hour']))
        days.append(dayMapper[item['day']])
        usernames.append(item['mentions'])
        scores.append(float(item['score']))
        authorFollowers.append(item['author_followers_count'])
        authorStatusCount.append(item['author_statuses_count'])
        authorFavoriteCount.append(item['author_favorite_count'])
        authorListedCount.append(item['author_listed_count'])
        authorIntervals.append(item['authorInterval'])
        parseLength.append(item['length'])
        headCount.append(item['head_count'])
        POScounts.append(POSRatio(item['pos_count']))
    inputFile.close()
    dataList = {'ids': ids, 'contents': contents, 'labels': labels, 'time': time, 'days': days, 'usernames': usernames, 'scores': scores, 'authorFollowers': authorFollowers,
                'authorStatusCount': authorStatusCount, 'authorFavoriteCount': authorFavoriteCount, 'authorListedCount': authorListedCount, 'authorIntervals': authorIntervals,
                'parseLength': parseLength, 'headCount': headCount, 'POScounts': POScounts}

    return dataList


def extractDecorations(dataList):
    mentionMapper = mapMention('dataset/commTweets/mention.json')
    afinn = Afinn()

    decorationFeatures = []
    for index, content in enumerate(dataList['contents']):
        temp = []
        words = word_tokenize(content)
        twLen = float(len(words))
        sentiScore = afinn.score(content)
        readScore = textstat.coleman_liau_index(content)
        temp.append(sentiScore / twLen)
        temp.append(twLen)
        temp.append(readScore)
        temp.append(dataList['parseLength'][index] / twLen)
        temp.append(dataList['headCount'][index] / twLen)
        temp.append(dataList['authorStatusCount'][index] / dataList['authorIntervals'][index])
        temp.append(dataList['authorFavoriteCount'][index] / dataList['authorStatusCount'][index])
        temp.append(dataList['authorListedCount'][index] / dataList['authorFollowers'][index])
        temp.append(dataList['days'][index])
        temp.append(dataList['time'][index])
        temp.append(1 if any(char.isdigit() for char in content) else 0)
        temp += dataList['POScounts'][index]
        # temp.append(content.count('URRL'))
        temp.append(1 if content.count('http://URL') > 0 else 0)
        # temp.append(content.count('HHTTG'))
        temp.append(1 if content.count('#HTG') > 0 else 0)
        # temp.append(content.count('USSERNM'))
        temp.append(1 if content.count('@URNM') > 0 else 0)
        # temp.append(content.count('!'))
        temp.append(1 if content.count('!') > 0 else 0)
        # temp.append(content.count('?'))
        temp.append(1 if content.count('?') > 0 else 0)
        mentionFlag = 0
        mentionFollowers = 0
        userCount = 0.0
        for user in dataList['usernames'][index]:
            if user in mentionMapper:
                userCount += 1
                if mentionMapper[user][0] == 1:
                    mentionFlag = 1
                mentionFollowers += mentionMapper[user][1]
        temp.append(mentionFlag)

        if userCount == 0:
            temp.append(0.0)
        else:
            temp.append(mentionFollowers / userCount)
        decorationFeatures.append(numpy.array(temp))

    return decorationFeatures


def tokenize(contents, tokenizerPath=None, outputPath=None):
    if tokenizerPath:
        tokenizer = pickle.load(open(tokenizerPath, "rb"))
    else:
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    processedContent = tokenizer.texts_to_sequences(contents)
    processedContent = pad_sequences(processedContent, maxlen=MAX_LEN)
    if outputPath:
        pickle.dump(tokenizer, open(outputPath, "wb"))

    return processedContent, tokenizer


def get_embeddings(tokenizer):
    print('Loading FastText embeddings...')
    def get_coefs(word, *arr):
        return word, numpy.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMB_PATH, encoding='utf-8'))
    word_index = tokenizer.word_index
    nb_words = len(word_index)
    embedding_matrix = numpy.zeros((nb_words + 1, EMB_SIZE))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix