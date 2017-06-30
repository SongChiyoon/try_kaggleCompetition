import tensorflow as tf
import numpy as np
import nltk
import pandas as pd
import csv

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

df['v1'] = df['v1'].map(lambda x:1 if x == 'ham' else 0)


df['words'] = df['v2'].map(lambda x: nltk.word_tokenize(x))

feature_set = zip(df['v1'], df['words'])


len_fSet = len(feature_set)

train_size = int(0.7 * len_fSet)
test_size = int(0.3 * len_fSet)

training_data = feature_set[:train_size]
test_data = feature_set[train_size:test_size]
print("training....")
classifier = nltk.NaiveBayesClassifier.train(training_data)

print("testing...")
print("accuracy : ",nltk.classify.accuracy(test_data))

