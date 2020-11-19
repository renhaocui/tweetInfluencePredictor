import utilities
import numpy
from scipy.sparse import hstack, csr_matrix
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Embedding, SpatialDropout1D, Reshape, Input, Concatenate, Bidirectional, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, max_error, mean_squared_error


filter_sizes = [1, 2, 3, 5]


def getFWN(input_dim):
    NN_model = Sequential()
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    #NN_model.summary()

    return NN_model


def getCNN(embedding_matrix):
    input = Input(shape=(utilities.MAX_LEN,))
    x = Embedding(embedding_matrix.shape[0], utilities.EMB_SIZE, weights=[embedding_matrix], trainable=True)(input)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((utilities.MAX_LEN, utilities.EMB_SIZE, 1))(x)
    mpool = []
    for f in filter_sizes:
        conv = Conv2D(8, kernel_size=(f, embedding_matrix.shape[1]), kernel_initializer='normal', activation='elu')(x)
        mpool.append(MaxPool2D(pool_size=(utilities.MAX_LEN - f + 1, 1))(conv))
    z = Concatenate(axis=1)(mpool)
    z = Flatten()(z)
    z = Dropout(0.2)(z)
    output = Dense(1, kernel_initializer='normal', activation="linear")(z)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model


def getLSTM(embedding_matrix):
    input = Input(shape=(utilities.MAX_LEN,))
    tweet_embedding = Embedding(embedding_matrix.shape[0], utilities.EMB_SIZE, weights=[embedding_matrix], trainable=True)(input)
    tweet_lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn'))(tweet_embedding)
    tweet_output = Dense(1, kernel_initializer='normal', activation="linear", name='output')(tweet_lstm)
    model = Model(inputs=input, outputs=tweet_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return model


def regressionEvaluate(predictions, test_labels):
    evs = explained_variance_score(test_labels, predictions)
    me = max_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)

    return evs, me, mse


def processCommTweets(dataFilename, resultFilename, modelPath, featureMode, embeddingMode):
    resultFile = open(resultFilename, 'a')

    dataList = utilities.loadData(dataFilename)
    decorationFeatures = utilities.extractDecorations(dataList)
    if embeddingMode > 0:
        embeddingFeature = utilities.getEmbeddings(embeddingMode, dataList['contents'], dataList['ids'])

    if featureMode < 0:
        features, tokenizer = utilities.tokenize(dataList['contents'], outputPath='dataset/commTweets/model/tokenizer.pkl')
        embedding_matrix = utilities.get_embeddings(tokenizer)
    if featureMode == 1:
        resultFile.write('decoration features only \n')
        features = decorationFeatures
    elif featureMode == 2:
        resultFile.write('decoration and embedding' + '\n')
        features = hstack((embeddingFeature, csr_matrix(numpy.array(decorationFeatures))), format='csr')
    elif featureMode == 3:
        resultFile.write('embedding only' + '\n')
        features = csr_matrix(embeddingFeature)

    labels = dataList['labels']
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

    print('Training...')
    if featureMode > 0:
        regression_model = getFWN(len(feature_train[0]))
    elif featureMode == -1:
        regression_model = getCNN(embedding_matrix)
    elif featureMode == -2:
        regression_model = getLSTM(embedding_matrix)
    callbacks = [
        EarlyStopping(
            min_delta=0,
            patience=5,
            restore_best_weights=True,
            monitor='val_loss',
            mode='auto',
            verbose=1)
    ]
    regression_model.fit(numpy.array(feature_train), numpy.array(label_train), epochs=500, batch_size=32, validation_split=0.1, callbacks=callbacks)
    regression_model.save(modelPath)

    print('Evaluating...')
    predictions = regression_model.predict(numpy.array(feature_test))
    evs, me, mse = regressionEvaluate(predictions[:, 0], label_test)

    print('Explained Variance Score: ', evs)
    print('Max Error: ', me)
    print('Mean Square Error: ', mse)

    resultFile.write('Explained Variance Score: ' + str(evs) + '\n')
    resultFile.write('Max Error: ' + str(me) + '\n')
    resultFile.write('Mean Square Error: ' + str(mse) + '\n')
    resultFile.close()


if __name__ == "__main__":
    processCommTweets('dataset/commTweets/features.json', 'dataset/commTweets/results.txt', 'dataset/commTweets/model/regression_1_0.model', 1, 0)
