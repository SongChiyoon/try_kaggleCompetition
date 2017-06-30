import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import tensorflow as tf
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
    for index in range(test.shape[0]):
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

for index in range(train.shape[0]):
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


X = tf.placeholder(tf.float32, shape=[None,6])
Y = tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([6,6]), name='weight1')
b1 = tf.Variable(tf.random_normal([6]), name = 'bias1')
layer1 = tf.nn.tanh(tf.matmul(X,w1)+b1)
w2 = tf.Variable(tf.random_normal([6,6]), name='weight2')
b2 = tf.Variable(tf.random_normal([6]), name = 'bias2')
layer2 = tf.nn.tanh(tf.matmul(layer1,w2)+b2)

w3 = tf.Variable(tf.random_normal([6,1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer1,w3)+b3)

train_x = np.array(train[:,[2,4,5,6,7,9]], dtype=float)
train_y = np.array(train[:,4],dtype=float)
train_y = np.reshape(train_y, (-1,1))



cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

print(train_x.shape, train_y.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        if step % 400 == 0:
          cost_val, _ = sess.run([cost, optimizer],feed_dict={X:train_x, Y:train_y})
          print(step, cost_val)

'''clf = linear_model.LogisticRegression()
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
'''
