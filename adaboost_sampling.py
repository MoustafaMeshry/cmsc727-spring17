"""
Adaboost on RNN for binary sequence classification
Dataset: IMDB sentiment dataset
"""
from __future__ import division, print_function, absolute_import

import os.path
import pickle
import sys
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tflearn

import numpy as np
import tensorflow as tf

def buildNetwork(numH, dropoutProb):
    tf.reset_default_graph()
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=numH)
    net = tflearn.lstm(net, numH, dropout=dropoutProb)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')
    return net

# ===========================================================================

def computeAccuracy(labels, gtLabels):
    acc = 100 * np.sum(labels == gtLabels) / len(labels)
    return acc

# ===========================================================================

def evaluateAdaboost(models, alphas, data, gtLabels, adaScores=None, printResults=False):
    if adaScores is None:
        n = len(gtLabels)
        adaScores = np.zeros(n * 2).reshape(n, 2) # 2 is the number of classes
    for i in range(len(alphas)):
        scores = predict(models[i], data)
        adaScores = adaScores + alphas[i] * scores
        if printResults:
            modelLabels = np.argmax(scores, 1)
            modelAcc = computeAccuracy(modelLabels, gtLabels)
            print("Model #" + str(i+1) + ": Accuracy = " + str(modelAcc))

    labels = np.argmax(adaScores, 1)
    acc = computeAccuracy(labels, gtLabels)
    if printResults:
        print("Adaboost Accuracy = " + str(acc))
        print("=============================================\n")

    return acc, adaScores

# ===========================================================================

def predict(model, data):
    n = len(data)
    y = np.zeros(n * 2).reshape(n, 2) # 2 is the number of classes
    chunkSize = 5000
    for i in range(0, n, chunkSize):
        endIndex = np.minimum(i + chunkSize, n)
        score = model.predict(data[i : endIndex, :])
        y[i : endIndex, :] = score

    return y

# ===========================================================================

def predictLabels(model, data):
    yScores = predict(model, data)
    yLabels = np.argmax(yScores, 1)
    return yLabels

# ===========================================================================

def printResultsSummary(resultsArr, header=None, metric="Results", entryLabel="iter",
                        fout=sys.stdout):
    if header:
        fout.write(header + "\n")
        fout.write("-" * len(header))
        fout.write("\n")

    for i in range(len(resultsArr)):
        fout.write(metric + " of " + entryLabel + "#" + str(i+1) + ": = " + str(
            resultsArr[i]) + "\n")

# ===========================================================================

def trainModel(net, trainX, trainY,  nEpochs=10, validationX=None, validationY=None):
    model = tflearn.DNN(net, tensorboard_verbose=0)
    if (validationX == None or validationY == None):
        model.fit(trainX, trainY, n_epoch=nEpochs, validation_set=0.1, show_metric=True, batch_size=32)
    else:
        model.fit(trainX, trainY, n_epoch=nEpochs, validation_set=(validationX, validationY), show_metric=True, batch_size=32)
    return model;

# ===========================================================================

def runIMDBExperiment(samplingRatio=0.5, nEpochs=5, nBoostIters=10,
        dropoutProb=0.8, numH=128):
    # Constants
    kDataDir = 'models'

    # Dictionary keys
    adaScoresTestKey = 'adaScoresTest'
    adaScoresTrainKey = 'adaScoresTrain'
    alphasKey = 'alphas'
    boostTestAccKey = 'boostTestAcc'
    boostTrainAccKey = 'boostTrainAcc'
    modelsTrainAccKey = 'modelsTrainAcc'
    modelsTestAccKey = 'modelsTestAcc'
    numSavedModelsKey = 'numSavedModels'
    wVecsKey = 'wVecs'


    # IMDB Dataset loading
    train, validation, test = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
    trainX, trainY = train
    validationX, validationY = validation
    testX, testY = test

    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    validationX = pad_sequences(validationX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    validationY = to_categorical(validationY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    trainLabels = np.argmax(trainY, 1)
    validationLabels = np.argmax(validationY, 1)
    testLabels = np.argmax(testY, 1)


    nTrain = len(trainY)
    w_boost = np.ones(nTrain) / nTrain
    sampleSz = int(samplingRatio * nTrain)
    models = [None] * nBoostIters
    alphas = np.ones(nBoostIters)

    numLoadedModels = 0
    metaDataFile = os.path.join("models", "modelsMeta.pckl")

    if not os.path.isfile(metaDataFile):
        metaData = dict()
        metaData[numSavedModelsKey] = 0
        metaData[alphasKey] = np.zeros(nBoostIters)
        metaData[modelsTrainAccKey] = np.zeros(nBoostIters)
        metaData[modelsTestAccKey] = np.zeros(nBoostIters)
        metaData[boostTrainAccKey] = np.zeros(nBoostIters)
        metaData[boostTestAccKey] = np.zeros(nBoostIters)
        metaData[wVecsKey] = [None] * nBoostIters
        metaData[adaScoresTrainKey] = np.zeros(len(trainLabels) * 2).reshape(
                            len(trainLabels), 2) # 2 = num of classes
        metaData[adaScoresTestKey] = np.zeros(len(testLabels) * 2).reshape(
                            len(testLabels), 2) # 2 = num of classes
    else:
        with open(metaDataFile, 'rb') as f:
            metaData = pickle.load(f)

        numLoadedModels = metaData[numSavedModelsKey]
        if len(metaData[alphasKey]) < nBoostIters:
            oldLen = len(metaData[alphasKey])
            tmp = metaData[alphasKey]
            metaData[alphasKey] = np.zeros(nBoostIters)
            metaData[alphasKey][0:oldLen] = tmp
            tmp = metaData[modelsTrainAccKey]
            metaData[modelsTrainAccKey] = np.zeros(nBoostIters)
            metaData[modelsTrainAccKey][0:oldLen] = tmp
            tmp = metaData[modelsTestAccKey]
            metaData[modelsTestAccKey] = np.zeros(nBoostIters)
            metaData[modelsTestAccKey][0:oldLen] = tmp
            tmp = metaData[boostTrainAccKey]
            metaData[boostTrainAccKey] = np.zeros(nBoostIters)
            metaData[boostTrainAccKey][0:oldLen] = tmp
            tmp = metaData[boostTestAccKey]
            metaData[boostTestAccKey] = np.zeros(nBoostIters)
            metaData[boostTestAccKey][0:oldLen] = tmp
            tmp = metaData[wVecsKey]
            metaData[wVecsKey] = [None] * nBoostIters
            metaData[wVecsKey][0:oldLen] = tmp

        w_boost = metaData[wVecsKey][numLoadedModels-1]
        print("Loading " + str(numLoadedModels) + " trained models...")
        for i in range(numLoadedModels):
            modelFileName = 'model_' + str(i) + '.tfl'
            modelFilePath = os.path.join(kDataDir, modelFileName)
            net = buildNetwork(numH, dropoutProb)
            currModel = tflearn.DNN(net, tensorboard_verbose=0)
            currModel.load(modelFilePath, weights_only=True)
            print("Loaded model #" + str(i+1) + ".")
            models[i] = currModel
            alphas[i] = metaData[alphasKey][i]
        print("Loaded " + str(numLoadedModels) + " trained models!")


    adaScoresTrain = metaData[adaScoresTrainKey]
    adaScoresTest = metaData[adaScoresTestKey]
    for i in range(numLoadedModels, nBoostIters):
    #    sample = np.random.randint(0, nTrain, sampleSz)
        wCumSum = np.cumsum(w_boost)
        sample = np.searchsorted(wCumSum, np.random.rand(sampleSz))
        sampleX = trainX[sample, :]
        sampleY = trainY[sample, :]
        sampleLables = trainLabels[sample]

        # Train model
        net = buildNetwork(numH, dropoutProb)
        model = trainModel(net, sampleX, sampleY, nEpochs, validationX, validationY)
        models[i] = model
        modelFileName = 'model_' + str(i) + '.tfl'
        modelFilePath = os.path.join(kDataDir, modelFileName)
        model.save(modelFilePath)

        # Compute alpha and update weights
        modelTrainLabels = predictLabels(model, trainX)
        correctMask = modelTrainLabels == trainLabels
        eps = np.sum(w_boost[np.logical_not(correctMask)])
        alpha = 0.5 * np.log((1-eps)/eps)
        alphas[i] = alpha
        w_boost[correctMask] = w_boost[correctMask] / (2 * (1 - eps))
        w_boost[np.logical_not(correctMask)] = w_boost[np.logical_not(correctMask)] / (2 * eps)

        # Compute metrics
        modelTrainAcc = computeAccuracy(modelTrainLabels, trainLabels)
        modelTestLabels = predictLabels(model, testX)
        modelTestAcc = computeAccuracy(modelTestLabels, testLabels)
        boostTrainAcc,adaScoresTrain = evaluateAdaboost(
            [models[i]], [alphas[i]], trainX, trainLabels, adaScores=adaScoresTrain)
        boostTestAcc,adaScoresTest = evaluateAdaboost(
            [models[i]], [alphas[i]], testX, testLabels, adaScores=adaScoresTest)

        # Save/(update saved) dictionay
        metaData[numSavedModelsKey] = metaData[numSavedModelsKey] + 1
        metaData[alphasKey][i] = alpha
        metaData[modelsTrainAccKey][i] = modelTrainAcc
        metaData[modelsTestAccKey][i] = modelTestAcc
        metaData[boostTrainAccKey][i] = boostTrainAcc
        metaData[boostTestAccKey][i] = boostTestAcc
        metaData[wVecsKey][i] = np.copy(w_boost)
        metaData[adaScoresTrainKey] = adaScoresTrain
        metaData[adaScoresTestKey] = adaScoresTest
        with open(metaDataFile, 'wb') as f:
            pickle.dump(metaData, f)

        # Print results
        printResultsSummary(metaData[modelsTrainAccKey][0:i+1], 'Models Train Accuracy:', 'Accuracy',
                            'model')
        print("--------------------------------------------------")
        printResultsSummary(metaData[boostTrainAccKey][0:i+1], 'Adaboost Train Accuracy:', 'Accuracy',
                            'Adaboost iter')
        print("--------------------------------------------------")
        printResultsSummary(metaData[modelsTestAccKey][0:i+1], 'Models Test Accuracy:', 'Accuracy',
                            'model')
        print("--------------------------------------------------")
        printResultsSummary(metaData[boostTestAccKey][0:i+1], 'Adaboost Test Accuracy:', 'Accuracy',
                            'Adaboost iter')
        print("==================================================\n")


    # Print overall summary (to stdout and to file)
    print("Run Summary:")
    print("Number classifiers = " + str(nBoostIters))
    print("Number of epochs = " + str(nEpochs))
    print("Sampling ratio = " + str(samplingRatio))
    print("Dropout = " + str(dropoutProb))
    printResultsSummary(metaData[modelsTrainAccKey], 'Models Train Accuracy:', 'Accuracy', 'model')
    print("--------------------------------------------------")
    printResultsSummary(metaData[boostTrainAccKey], 'Adaboost Train Accuracy:', 'Accuracy',
                        'Adaboost iter')
    print("--------------------------------------------------")
    printResultsSummary(metaData[modelsTestAccKey], 'Models Test Accuracy:', 'Accuracy', 'model')
    print("--------------------------------------------------")
    printResultsSummary(metaData[boostTestAccKey], 'Adaboost Test Accuracy:', 'Accuracy',
                        'Adaboost iter')
    print("==================================================\n")

    resultsFileName = "results-iters=" + str(nBoostIters) + "-epochs=" + str(
        nEpochs) + "-sample=" + str(samplingRatio) + "-dropout=" + str(dropoutProb) + ".txt"
    resultsFilePath = os.path.join(kDataDir, resultsFileName)
    with open(resultsFilePath, 'w') as f:
        f.write("Run Summary:\n")
        f.write("Number classifiers = " + str(nBoostIters) + "\n")
        f.write("Number of epochs = " + str(nEpochs) + "\n")
        f.write("Sampling ratio = " + str(samplingRatio) + "\n")
        f.write("Dropout = " + str(dropoutProb) + "\n")
        printResultsSummary(metaData[modelsTrainAccKey], 'Models Train Accuracy:',
                            'Accuracy', 'model', f)
        f.write("--------------------------------------------------\n")
        printResultsSummary(metaData[boostTrainAccKey], 'Adaboost Train Accuracy:', 'Accuracy',
                            'Adaboost iter', f)
        f.write("--------------------------------------------------\n")
        printResultsSummary(metaData[modelsTestAccKey], 'Models Test Accuracy:', 'Accuracy', 'model', f)
        f.write("--------------------------------------------------\n")
        printResultsSummary(metaData[boostTestAccKey], 'Adaboost Test Accuracy:', 'Accuracy',
                            'Adaboost iter', f)
        f.write("==================================================\n")


