from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from string import ascii_lowercase
import argparse
import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Define one-hot encoding dictionary
one_hot_encoding = {}
for i, l in enumerate(ascii_lowercase):
    bits = [0] *26
    bits[i] = 1
    one_hot_encoding[l] = bits

one_hot_decoding = {}
for i, l in enumerate(ascii_lowercase):
    one_hot_decoding[i] = l


# One-hot encode a numpy array of letters
def one_hot_encode(y):
    y_encoded = np.empty([y.shape[0], 26])
    for i, l in np.ndenumerate(y):
        y_encoded[i, :] = one_hot_encoding[l]
    return y_encoded

# One hot decoding
def one_hot_decode(y):
    y_decoded = []
    for l in y:
        l = l.tolist()
        m = max(l)
        for i, v in enumerate(l):
            if v == m:
                index = i
        y_decoded.append(one_hot_decoding[index])
    return y_decoded


def next_batch(size, x, y):
    # Return a total of `num` samples from the array `data`.#
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)
    idx = idx[0 : size-1]
    x_shuffle = np.asarray([x[i, :] for i in idx])
    y_shuffle = np.asarray([y[i, :] for i in idx])
    return x_shuffle, y_shuffle


def main(_):
    # Read training samples and labels from the csv file
    data = np.genfromtxt('data/train.csv', delimiter=',',
                         skip_header=1, dtype='S10')
    x = data[:, 4:].astype(np.float)
    y = data[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0)

    # One-hot encode the training labels
    y_train_encoded = one_hot_encode(y_train)

    # One-hot encode the test labels
    y_test_encoded = one_hot_encode(y_test)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 128])
    y_ = tf.placeholder(tf.float32, [None, 26])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 16, 8, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([4 * 2 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 26])
    b_fc2 = bias_variable([26])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # define loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # define training operation
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    for i in range(100000):

        x_train_batch, y_train_batch = next_batch(50, x_train, y_train_encoded)

        if i % 100 == 0:
            test_accuracy = accuracy.eval(feed_dict={
                x: x_test, y_: y_test_encoded, keep_prob: 1.0})

            print("step %d, testing accuracy %g" % (i, test_accuracy))

        train_step.run(feed_dict={x: x_train_batch, y_: y_train_batch, keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: x_test, y_: y_test_encoded, keep_prob: 1.0}))

    # Read testing data
    data = np.genfromtxt('data/test.csv', delimiter=',',
                         skip_header=1, dtype='S10')
    x_test = data[:, 4:].astype(np.float)
    ids = data[:, 0]

    # Make predicitons
    result = y_conv.eval(feed_dict={x: x_test, keep_prob: 1.0})
    result_decoded = one_hot_decode(result)

    # write the result to a file
    f = open('result.csv', 'w')
    f.write("Id,Prediction\n")

    for i, v in enumerate(result_decoded):
        f.write(ids[i] + ',' + v + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
