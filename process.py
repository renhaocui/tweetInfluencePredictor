__author__ = 'rencui'
import tweetBlender
import dataLabeler
import contenter
import parserExtractor
import tweetGrouper
import runModel
import usernameExtractor

# blender the brand dataset
tweetBlender.blend(300, 0)
# remove outliers and assign labels
dataLabeler.label(2)
# extract content and parse
contenter.contenterExtractor()
# need to run TweeboParser here
parserExtractor.extractor()
# grouping the data
tweetGrouper.totalGrouper()
#collect meta data
usernameExtractor.collector()
# run the experiment
runModel.runModel(1, 'totalGroup', 2, 4, 'SVM', outputFile='results/temp.result')