# Test MNIST task
# We used this file to test our network on the MNIST task


# Execute this file while being in the MNIST directory (not from parent directory e.g.)

# Import modules
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import the network class from .py file in parent directory
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residualCNN import RESNET


# Specify task and build network
net = RESNET(task='MNIST',direc='below')
net.network()


### ONLY FOR TESTING, NOT FOR RETRAINING (variables not stored)

with tf.Session() as session:

    saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./weightsMNIST.ckpt") 


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
        p.append(net.accuracy.eval(feed_dict = {net.x: trainImgs[k*step:(k+1)*step], 
            net.y_:mnist.train.labels[k*step:(k+1)*step], net.keep_prob:1.0}))

    print()
    print('Train Accuracy MNIST = ', np.mean(p))


    # Same for evaluation  set
    size = mnist.validation.images.shape[0]
    testImgs = np.empty([size,28,28,3])
    for i in range(3):
        testImgs[:,:,:,i] = mnist.validation.images.reshape([size,28,28])
    print('Validation Accuracy MNIST ', net.accuracy.eval(feed_dict = {net.x: testImgs, 
        net.y_: mnist.validation.labels, net.keep_prob:1.0}))


    # Same for test set
    size = mnist.test.images.shape[0]
    testImgs = np.empty([size,28,28,3])
    for i in range(3):
        testImgs[:,:,:,i] = mnist.test.images.reshape([size,28,28])
    print('Test Accuracy MNIST ', net.accuracy.eval(feed_dict = {net.x: testImgs, 
        net.y_: mnist.test.labels, net.keep_prob:1.0}))
    print()










