import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import numpy as np
import csv
def modiTest(test):
    age = test[:,4]
    sib = test[:, 5]
    parch = test[:, 6]
    fare = test[:, 8]
    ageMean = np.mean(np.float32(age[age[:] != ""]))
    sibMean = np.mean(np.float32(sib[sib[:] != ""]))
    parchMean = np.mean(np.float32(parch[parch[:] != ""]))
    fareMean = np.mean(np.float32(fare[fare[:] != ""]))
    for index in xrange(test.shape[0]):
        if test[index][3] == 'male':
            test[index][3] = 0
        else:
            test[index][3] = 1
        if test[index][4] == "":
            test[index][4] = ageMean
        if test[index][5] == "":
            test[index][5] = sibMean
        if test[index][6] == "":
            test[index][6] = parchMean
        if test[index][8] == "":
            test[index][8] = fareMean
    return test

with open("train.csv", 'r') as f:
    train = list(csv.reader(f, delimiter=","))
with open("test.csv", 'r') as f:
    test = list(csv.reader(f, delimiter = ","))

test = test[1:]

#index
isex = 4
iage = 5
isib = 6
iparch = 7
ifare = 9

#for index in xrange()
train = np.array(train[1:], dtype=None)

age = train[:,iage]
sib = train[:,isib]
parch = train[:,iparch]
fare = train[:,ifare]
ageMean = np.mean(np.float32(age[age[:] != ""]))
sibMean = np.mean(np.float32(sib[sib[:] != ""]))
parchMean = np.mean(np.float32(parch[parch[:] != ""]))
fareMean = np.mean(np.float32(fare[fare[:] != ""]))

for index in xrange(train.shape[0]):
    if train[index][isex] == 'male':
        train[index][isex] = 0
    else:
        train[index][isex] = 1
    if train[index][iage] == "":
        train[index][iage] = ageMean
    if train[index][isib] == "":
        train[index][isib] = sibMean
    if train[index][iparch] == "":
        train[index][iparch] = parchMean
    if train[index][ifare] == "":
        train[index][ifare] = fareMean


train_x = np.array(train[:,[2,4,5,6,7,9]], dtype=float)
train_y = np.array(train[:,4],dtype=float)
#train_y = np.reshape(train_y, (-1,1))

df = pd.read_csv('train.csv')
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]



iris = datasets.load_iris()

clf = linear_model.LogisticRegression()
clf.fit(train_x,train_y)

test = np.array(test, dtype=None)
test = modiTest(test)


test_x = np.array(test[:,[1,3,4,5,6,8]], dtype=float)


result = np.array(clf.predict(test_x))
print result
id = [[i+892] for i in xrange(0,len(result))]

result = np.reshape(result, (-1,1))
result = np.hstack((id,result))



np.savetxt("result.csv", result, delimiter=",", fmt='%1d')

compare = np.loadtxt("gender_submission.csv", delimiter=",",dtype=float)

correct = 0
for i in xrange(compare.shape[0]):
    if compare[i][1] == result[i][1]:
        correct += 1

print correct