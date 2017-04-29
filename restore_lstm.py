# Code to restore a saved model of lstm and run it

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tensorflow as tf
import numpy as np

def inference(model,X,Y, remark=None):
  
  if remark:
    print(remark)
  labels_prob = model.predict(X)
  # Obtain labels from predicted probabilities
  my_labels = np.argmax(labels_prob,axis=1)
  return my_labels


def accuracy(Y, my_labels, remark=None):
  if remark:
    print(remark)
  actual_labels = np.argmax(Y,axis=1)
  correct = np.sum(np.equal(actual_labels,my_labels))
  print('Accuracy after voting = %f'%(correct/len(Y))) # This is the accuracy from labels predicted from probabilities
  

# IMDB Dataset loading
train, val, test = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.2)
trainX, trainY = train
valX, valY = val
testX, testY = test


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
valX = pad_sequences(valX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
valY = to_categorical(valY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=64)
net = tflearn.lstm(net, 64, dropout=0.8) # edited
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.005, loss='categorical_crossentropy')

N=7 # No. of classifiers
majority=1  # Whether we are employing majority voting or checking one single classifier
if majority:
  models = [tflearn.DNN(net) for i in range(N)] # Create space for N models
  models[0].load("/tmp/tflearn_logs0/8250") # Loads a model with the given name (cannot be pre-determined)
  models[1].load("/tmp/tflearn_logs1/8282")
  models[2].load("/tmp/tflearn_logs2/8232")
  models[3].load("/tmp/tflearn_logs3/8184")
  models[4].load("/tmp/tflearn_logs4/8216")
  models[5].load("/tmp/tflearn_logs5/8254")
  models[6].load("/tmp/tflearn_logs6/8208")
  for i in range(N):
    score_val = models[i].evaluate(valX,valY)  # Evaluate all the models
    score_test = models[i].evaluate(testX,testY)
    print('Accuracy of classifier %d on validation set = %f and test set = %f'%(i+1,score_val[0],score_test[0]))
else:
  # Used to load and evaluate a single model
  model = tflearn.DNN(net)
  model.load("/tmp/tflearn_logs4/8072")
  score_val = model.evaluate(valX,valY)
  score_test = model.evaluate(testX,testY)
  print('Accuracy on validation set = %f and test set = %f'%(score_val[0],score_test[0]))

# For majority voting
if majority:
  my_labels = [i for i in range(N)]
  for i in range(N):
    my_labels[i] = inference(models[i],testX,testY) # Obtain the labels of all the models

  # Now vote and check accuracy
  total_labels = np.zeros(len(testY)); final_labels = np.zeros(len(testY))
  for i in range(len(testY)):
     for j in range(N):
       total_labels[i] = total_labels[i] + (my_labels[j])[i]
     final_labels[i] = np.round(total_labels[i]/N)

  accuracy(testY, final_labels, 'Employing majority voting on test set...')
