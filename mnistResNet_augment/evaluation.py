# This file is to test the performance of our Deep Residual CNN on MNIST (as a proof for correct implementation)

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.protobuf import saver_pb2
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Change this if necessary
pathToWeights = './'



###### NETWORK ##########

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, strideX = 1, strideY = 1):
  #  Convolution and batch normalization
  conv = tf.nn.conv2d(x, W, strides=[1, strideX, strideY, 1], padding='SAME')
  mean, var = tf.nn.moments(conv, [0, 1, 2])
  activation = tf.nn.batch_normalization(conv, mean, var, None, None, 1e-3)
  return activation

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)


# 6 convolutions - 64 kernels (3x3)
W_conv1 = weight_variable([3,3,3,64])
b_conv1 = bias_variable([64])

# Feature map size is halved by stride of 2
activation = conv2d(x, W_conv1, 2, 2) + b_conv1
h_conv1 = tf.nn.relu(activation)

W_conv2 = weight_variable([3,3,64,64])
b_conv2 = bias_variable([64])
activation = conv2d(h_conv1, W_conv2, 1, 1) + b_conv2
h_conv2 = tf.nn.relu(activation)

W_conv3 = weight_variable([3,3,64,64])
b_conv3 = bias_variable([64])
activation = conv2d(h_conv2, W_conv3, 1, 1) + b_conv3
residual = activation + h_conv1
h_conv3 = tf.nn.relu(activation)

W_conv4 = weight_variable([3,3,64,64])
b_conv4 = bias_variable([64])
activation = conv2d(h_conv3, W_conv4, 1, 1) + b_conv4
h_conv4 = tf.nn.relu(activation)

W_conv5 = weight_variable([3,3,64,64])
b_conv5 = bias_variable([64])
activation = conv2d(h_conv4, W_conv5, 1, 1) + b_conv5
h_conv5 = tf.nn.relu(activation)

W_conv6 = weight_variable([3,3,64,64])
b_conv6 = bias_variable([64])
activation = conv2d(h_conv5, W_conv6, 1, 1) + b_conv6
residual = activation + h_conv4
h_conv6 = tf.nn.relu(activation)

# 3 convolutions - 128 kernels (3x3)
W_conv7 = weight_variable([3,3,64,128])
b_conv7 = bias_variable([128])

# Halve size of map features
activation = conv2d(h_conv6, W_conv7, 2, 2) + b_conv7
h_conv7 = tf.nn.relu(activation)

W_conv8 = weight_variable([3,3,128,128])
b_conv8 = bias_variable([128])
activation = conv2d(h_conv7, W_conv8, 1, 1) + b_conv8
h_conv8 = tf.nn.relu(activation)

W_conv9 = weight_variable([3,3,128,128])
b_conv9 = bias_variable([128])
activation = conv2d(h_conv8, W_conv9, 1, 1) + b_conv9
residual = activation + h_conv7
h_conv9 = tf.nn.relu(activation)



W_fc1 = weight_variable([7*7*128, 1024])

b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_conv8, [-1, 7*7*128])



h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
OUT = tf.nn.softmax(y_conv)

# Output Functions

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

######################## END NETWORK #######################






with tf.Session() as session:

    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
    # Requires files resnet.ckpt-NUM and resnet.ckp-NUM.meta and NOT MORE FILES
    saver.restore(session, pathToWeights+"weights.ckpt")

    # Evaluate Training Accuracy. Memory Error with single batch of size 55000, therefore we use 5 batches of size 11000
    # If that is still do much for your system, set the splitter variable accordingly (i.e. 55000%splitter == 0 --> TRUE) 

    splitter = 11    
    size = mnist.train.images.shape[0]
    step = size // splitter

    # Read in images and extend to 3D (we work with color images)
    trainImgs = np.empty([size,28,28,3])
    for k in range(3):
        trainImgs[:,:,:,k] = mnist.train.images.reshape([size,28,28])

    # Now check performance on train set
    p = []
    for k in range(splitter):
        p.append(accuracy.eval(feed_dict = {x: trainImgs[k*step:(k+1)*step], y_:mnist.train.labels[k*step:(k+1)*step], keep_prob:1.0}))
    print()
    print('Train Accuracy MNIST = ', np.mean(p))

    # Same for test set
    size = mnist.test.images.shape[0]
    testImgs = np.empty([size,28,28,3])
    for i in range(3):
        testImgs[:,:,:,i] = mnist.test.images.reshape([size,28,28])
    print('Test Accuracy MNIST ', accuracy.eval(feed_dict = {x: testImgs, y_: mnist.test.labels,keep_prob:1.0}))



