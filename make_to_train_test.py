#!/usr/bin/env python

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
def process_tweets_to_train_test(tweets, sentiment, tokenizer, n_labels, prop_dataset=0.9, max_len=15):
    
    # encode the tweets
    encoded = np.array(tokenizer.texts_to_sequences(tweets))
    encoded_pre_pad = pad_sequences(encoded, padding='pre',truncating = 'post')

    
    ## define training size etc
    len_dataset = len(tweets)
    n_datapoints = int(len_dataset * prop_dataset) # number of training data 
    
    
    # separate training and test sets
    train_x = tf.convert_to_tensor(encoded_pre_pad[:n_datapoints])
    test_x = tf.convert_to_tensor(encoded_pre_pad[n_datapoints:])
    train_y = tf.convert_to_tensor(sentiment[:n_datapoints])
    test_y = tf.convert_to_tensor(sentiment[n_datapoints:])
    
    # one hot encode the labels
    train_y += 1
    test_y += 1
    train_y = tf.one_hot(train_y, n_labels)
    test_y = tf.one_hot(test_y, n_labels)
    
    return train_x, train_y, test_x, test_y