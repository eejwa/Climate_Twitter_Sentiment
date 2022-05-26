#!/usr/bin/env python

import re
from spellchecker import SpellChecker
import spacy 
import random
import numpy as np 

spell_corrector = SpellChecker()

def remove_label_values(df, values):
    """
    Parameters
    ----------
    df : pandas dataframe 
       : dataframe to have values removed 
       
    values : list of integers 
           : labels to be removed from dataframe df
    Returns
    -------
    
    df : pandas dataframe
       : dataframe with the rows containing values removed. 
    
    """
    return df[df.sentiment.isin(values) == False]

def spell_correction(text):
    """
    
    Paramters
    ---------
    text : string
         : text to be spell checked and corrected.
    
    Returns
    -------
    
    correct_spelling : string
         : text will spelling mistakes removed.
    """
    # initialize empty list to save correct spell words
    correct_words = []
    # extract spelling incorrect words by using unknown function of spellchecker
    misSpelled_words = spell_corrector.unknown(text.split())

    for each_word in text.split():
        if each_word in misSpelled_words:
            right_word = spell_corrector.correction(each_word)
            correct_words.append(right_word)
            
        else:
            correct_words.append(each_word)

    # joining correct_words list into single string
    correct_spelling = ' '.join(correct_words)
    return correct_spelling

def preprocessing(tweets):
    """
    
    Parameters
    ----------
    tweets : list of strings
           : list of tweets to be processed
               
    Returns
    -------
    processed_tweets : list of strings
                     : tweets shuffled with urls, 
                       htmls, numbers, stop words, 
                       punctuation and retweets
                       removed and in lower case.
    
    """
    # remove links, @ and #
    html_pattern = r'<.*?>'
    url_pattern = r'https?://\S+'
    number_pattern = r'\d+'

    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words

    # shuffle tweets and sentiments
    p = np.random.permutation(len(tweets))

    processed_tweets = tweets[p]

    for i,tweet in enumerate(processed_tweets):
        # convert to lower case
        tweet = tweet.lower()

        # remove urls
        tweet = re.sub(pattern=url_pattern, repl="", string=tweet)

        # remove retweets
        tweet = re.sub(r"RT", "", tweet)
        tweet = re.sub(r"rt", "", tweet)

        # remove punctuation 
        tweet = re.sub(r'[^\w\s]', '', tweet)

        # remove htmls
        tweet = re.sub(pattern=html_pattern, repl='', string=tweet)

        # remove numbers
        tweet = re.sub(pattern=number_pattern, repl="", string=tweet)

        # spelling correction 
        # very slow so not using
        # tweet = spell_correction(tweet)

        # remove stop words
        tweet = " ".join([w for w in tweet.split() if not w in stopwords])

        # remove strange words/characters
        # again to save time on my laptop, im not doing this...
        # doc = nlp(tweet)
        # tweet = " ".join([token.text for token in doc if token.is_alpha])

        # Lemmatisation
        # Is also slow so not doing it on my laptop...   
        # doc = nlp(tweet)
        # tweet = " ".join([token.lemma_ for token in doc])

        processed_tweets[i] = tweet

    return processed_tweets