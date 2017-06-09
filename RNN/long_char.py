import tensorflow as tf

story = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
data = list(set(story))



indexToChar = {c : i for i,c in enumerate(data)}

sequence_len = 10
hidden_size = len(data)
num_classess = len(data)
X_set = []
Y_set = []
for i in range(0, len(story) - sequence_len):
    x_str = story[i:i+sequence_len]
    y_str = story[i+1:i+1+sequence_len]
    x_data = [indexToChar[index] for index in x_str]
    y_data = [indexToChar[index] for index in y_str]

    X_set.append(x_data)
    Y_set.append(y_data)

X = tf.placeholder(tf.float32, [None, sequence_len])
Y = tf.placeholder(tf.float32, [None, sequence_len])

'''X = tf.one_hot(X, num_classess)

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
initial_state = cell.zeros(1, tf.float32)
outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)'''

