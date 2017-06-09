import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


data = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
data = data[::-1]
data = MinMaxScaler(data)
x = data
y = data[:,[-1]]

sequence = 7
inputD = 5
outD = 1
x_data = []
y_data = []
for i in range(0, len(x) - sequence):
    _x = x[i:i+sequence]
    _y = y[i+sequence]

    x_data.append(_x)
    y_data.append(_y)
x_data = np.array(x_data)
y_data = np.array(y_data)

X = tf.placeholder(tf.float32, [None, None, inputD])
Y = tf.placeholder(tf.float32, [None, outD])

cell = rnn.BasicLSTMCell(num_units=10, state_is_tuple=True, activation=tf.tanh)

outputs, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], outD)


sess = tf.Session()
new_saver = tf.train.import_meta_graph('stock_prediction_Model-1000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

prediction = sess.run(y_pred, feed_dict={X:x_data})

plt.plot(y_data, 'red')
plt.plot(prediction)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()