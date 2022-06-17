# Sentiment Analysis of Climate Related Tweets

This notebook and associated python files takes 44k tweets which have been labeled as positive, negative, neutral or news, preprocesses them and categorises them. It became clear that the performance of the networks heavily depended on the dataset and labels used. It is very easy to distinguish between positive and negative tweets but the neutral and news tweets make things challenging. 

## Data

The data is not provided in this repository but is available here: https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset

## Files

- Sentiment_Analysis.ipynb: jupyter notebook where the processing and training happens. Visualisation of the dataset and results are also provided. 

- preprocessing_functions.py: used in the notebook, functions to preprocess the tweets. 

- plotting_functions.py: functions to visualise the dataset and the training. 

- make_to-train_test.py: encode and split the dataset into usable training and validation sets. 

## Use

To use this notebook and files, you need to download the data from the link above and install the anaconda environment provided using: 

`conda env create -f environment.yml`




