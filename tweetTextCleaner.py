import re
from ekphrasis.classes.segmenter import Segmenter
seg_tw = Segmenter(corpus="twitter")

__author__ = 'rencui'

charList = ['&gt;', '&amp;', '|', '&lt;3', '()', 'amp']


def removeEmoji(input, token):
    emojis = re.findall(r'\\u....', input)
    if len(emojis) != 0:
        for char in emojis:
            input = input.replace(char, token)
    return input


def shrinkPuncuation(input):
    input = re.sub('\.+', '.', input)
    input = re.sub(',+', ',', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('=+', '=', input)
    input = re.sub('-+', '-', input)
    input = re.sub('_+', '_', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('\s+', ' ', input)
    return input


def removeUsername(input, token):
    users = re.findall(r'@(\w+)', input)
    if len(users) != 0:
        for user in users:
            input = input.replace('@'+user, token)
    return input


def tokenizeLinks(input, token):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, token)
    return input


def removeHashtag(input, token, segment=False):
    hts = re.findall(r'#(\w+)', input)
    if len(hts) != 0:
        for ht in hts:
            if segment:
                expandedHT = seg_tw.segment(ht)
                input = input.replace('#'+ht, expandedHT)
            else:
                input = input.replace(ht, token)
    return input


def tweetCleaner(input, segment=False):
    input = input.replace('w/', 'with')
    input = input.replace('w/o', 'without')
    input = removeUsername(input, '')
    input = removeHashtag(input, 'HTG', segment)
    input = removeEmoji(input, '')
    #input = tokenizeLinks(input, 'http://URL')
    input = tokenizeLinks(input, '')
    for char in charList:
        input = input.replace(char, '')
    input = input.replace('\\"', '"')
    input = input.replace('\\', '')
    input = shrinkPuncuation(input)
    if input != '':
        if input[0] == ' ':
            input = input[1:]
    return input
