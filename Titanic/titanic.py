import tensorflow as tf
import numpy as np

import csv
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

with open("train.csv", 'r') as f:
    train = list(csv.reader(f, delimiter=","))
with open("test.csv", 'r') as f:
    test = list(csv.reader(f, delimiter = ","))

train = train[1:]
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


train_x = np.array(train[:,[2,4,5,6,7,9]], dtype=float)
train_y = np.array(train[:,4],dtype=float)
train_y = np.reshape(train_y, (-1,1))

print(train_x.shape, train_y.shape)

X = tf.placeholder(tf.float32,shape = [None, 8])
Y = tf.placeholder(tf.float32,shape = [None,1])

w = tf.Variable(tf.random_normal([8,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
###
b = tf.Print(b, [b], "Bias: ")
w = tf.Print(w, [w], "Weight: ")
X = tf.Print(X, [X], "TF_in: ")
matmul_result = tf.matmul(X, w)
matmul_result = tf.Print(matmul_result, [matmul_result], "Matmul: ")
hypothesis = tf.nn.sigmoid(matmul_result+b)
####
#hypothesis = tf.nn.sigmoid(tf.matmul(X,w)+b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print "step: ", step
            print "cost_val", cost_val

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


