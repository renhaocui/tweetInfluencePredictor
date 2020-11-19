from keras.models import load_model
import utilities
import numpy


def predictLabel(modelDir, testFilename, tokenizerPath, featureMode):
    regression_model = load_model(modelDir)
    dataList = utilities.loadData(testFilename)

    if featureMode < 0:
        features, tokenizer = utilities.tokenize(dataList['contents'], tokenizerPath)
        embedding_matrix = utilities.get_embeddings(tokenizer)
    elif featureMode == 1:
        decorationFeatures = utilities.extractDecorations(dataList)
        features = decorationFeatures

    predictions = regression_model(numpy.array(features))
    for score in numpy.array(predictions).tolist():
        print(score[0])


if __name__ == '__main__':
    predictLabel('dataset/commTweets/model/regression_1_0.model', 'dataset/commTweets/paraphrase_feature.json', 'dataset/commTweets/model/tokenizer.pkl', 1)
