#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    labels = ['1','2','0','-1']
    counts = df['sentiment'].value_counts().to_numpy()
    counts = counts[::-1]
    labels = labels[::-1]

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    bar = ax.bar(labels, counts, color='C1')

    # Add annotation to bars
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_xlabel('Sentiment label')
    ax.set_ylabel('Number of labels')
    ax.set_title('Distribution of Labels')

    plt.tight_layout()
    plt.show()
    return


def plot_word_frequency(top_N_words, top_N_word_frequency):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    ax.barh(top_N_words, top_N_word_frequency, color='C1')
    ax.invert_yaxis()
    ax.set_xlabel('word frequency', fontsize = 16)
    ax.set_ylabel('word', fontsize = 16)

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+50, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10,
                 color ='black')
    plt.tight_layout()
    plt.show()
    return

## plot the progress of training the two NN
def plot_history(history):
    """
    
    Parameters
    ----------
    history : keras history
            : history output from model.fit()
            
    Returns
    -------
    Nothing
    
    """
    epochs = range(1,6)
    
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(121)
                     
    ax1.plot(epochs, history.history['accuracy'], label='accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='val_accuracy')
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(loc='best')
    
    ax2 = fig.add_subplot(122)
    
    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(loc='best')

    
    plt.tight_layout()
    plt.show()

