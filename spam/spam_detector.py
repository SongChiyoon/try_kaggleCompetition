import tensorflow as tf
import numpy as np
import nltk
import pandas as pd
import csv

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

df['v1'] = df['v1'].map(lambda x:1 if x == 'ham' else 0)


df['words'] = df['v2'].map(lambda x: nltk.word_tokenize(x))

feature_set = zip(df['v1'], df['words'])
print(feature_set[1:5])