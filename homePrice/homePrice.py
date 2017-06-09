import pandas as pd

train = pd.read_csv('train.csv',delimiter=",")

print train.describe()