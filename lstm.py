# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tensorflow as tf
import numpy as np

N=7 # No. of classifiers
max_epochs = 5 # Max. no. of iterations
batch_size = 25
check_all = 0  # Whether or not model should be tested at the end of each epoch
valid_size=0.2
hidden_units = 64
eta=0.005 # This is the learning rate
dropout=0.8

# function to obtain labels form a model
def inference(model,X,Y, remark=None):
  
  if remark:
    print(remark)
  #score = model.evaluate(X,Y)
  #print('Accuracy from evaluate() function= %f'%(score[0]))
  labels_prob = model.predict(X)

  # Obtain labels from predicted probabilities
  my_labels = np.argmax(labels_prob,axis=1)
  return my_labels

# function to determine accuracy from the labels
def accuracy(Y, my_labels, remark=None):
  if remark:
    print(remark)
  actual_labels = np.argmax(Y,axis=1)
  correct = np.sum(np.equal(actual_labels,my_labels))
  print('Accuracy from predict() function = %f'%(correct/len(Y))) # This is the accuracy from labels predicted from probabilities
  
# function to evaluate a model on all the datasets
def model_eval(model):
  score1 = model.evaluate(trainX,trainY)
  score2 = model.evaluate(valX,valY)
  score3 = model.evaluate(testX,testY)
  print('Accuracy on training, validation and test set = %f, %f, %f'%(score1[0],score2[0],score3[0]))

# IMDB Dataset loading
train, val, test = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=valid_size)
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
net = tflearn.embedding(net, input_dim=10000, output_dim=hidden_units)
net = tflearn.lstm(net, hidden_units, dropout=dropout) 
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=eta, loss='categorical_crossentropy')

# Create duplicates of network for the classifiers
nets = [net for i in range(N)]; models = [i for i in range(N)] # dummy initialization for models array

# Training
for i in range(N):
  dir_save = "/tmp/tflearn_logs"+str(i)+'/'  # directory where models should be saved
  models[i] = tflearn.DNN(nets[i], tensorboard_verbose=0,tensorboard_dir=dir_save,checkpoint_path=dir_save, best_checkpoint_path=dir_save) 
  print('Training classifier %d...'%(i+1))
  models[i].fit(trainX, trainY, n_epoch=max_epochs, validation_set=(valX, valY), show_metric=True, batch_size=batch_size)

# Now test on validation set
my_labels = [i for i in range(N)]  # my_labels holds the labels for all the classifiers (dummy initialization)
for i in range(N):
  print('Testing classifier %d'%(i+1), end = ' ')
  my_labels[i] = inference(models[i],valX,valY,'On validation set')
  accuracy(valY, my_labels[i])

# Now test on test set
for i in range(N):
  print('Testing classifier %d'%(i+1), end = ' ')
  my_labels[i] = inference(models[i],testX,testY,'On test set')
  accuracy(testY, my_labels[i])

# Now let all classifiers vote and see the results
total_labels = np.zeros(len(testY)); final_labels = np.zeros(len(testY))
for i in range(len(testY)):
   for j in range(N):
     total_labels[i] = total_labels[i] + (my_labels[j])[i]
   final_labels[i] = np.round(total_labels[i]/N)   # Final labels are always 0 or 1
 
accuracy(testY, final_labels, 'Employing majority voting on test set')

#Now that we are done with training and testing, restore all the saved models and check it
if check_all :
  print('Restoring all the checkpoints of the models...')
  for i in range(max_epochs):
    modID = (int)(len(trainY)/batch_size)*(i+1)
    mod_path = '/tmp/tflearn_logs/-'+str(modID)
    models[0].load(mod_path)
    print('Saved model {}:'.format(i+1), end=' ')
    model_eval(models[0])





