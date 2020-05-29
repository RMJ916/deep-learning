#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

with open('Dataset/train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('Dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f)
with open('Dataset/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)
with open('Dataset/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('Dataset/', one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels


train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)
train_labels = train_labels.reshape(60000,1)
test_labels = test_labels.reshape(10000,1)

train_labels = np.dot(train_labels,np.zeros([1,1]))
test_labels = np.dot(test_labels,np.zeros([1,1]))

# X is our placeholder value for our input
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None,10])


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# Create an InteractiveSession 
sess = tf.InteractiveSession()

# Create an operation to initialize the variables we created
tf.global_variables_initializer().run()

for i in range(100):
    x_train, y_train = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))