# Test Identity task 
# We used this file to test our network on the ability to differentiate the identity of 10 Hollywood celebrities
# The task cannot be properly solved.


# Execute this file while being in the directory (not from parent directory e.g.)

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np

# Import the network class from .py file in parent directory
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residualCNN import RESNET


# Specify task and build network
net = RESNET(task='IDENTITY',direc='below')
net.network()


### ONLY FOR TESTING, NOT FOR RETRAINING (variables not stored)

with tf.Session() as session:

    saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./weightsIdentity.ckpt") 

    # Performance on training data
    Inds = np.random.permutation(net.numTrainImgs)
    trainIms, trainLabs = net.createBatch(Inds, 'TrainData')

    print('Training accuracy with dropout = ', net.accuracy.eval(feed_dict={net.x:trainIms, net.y_:trainLabs, net.keep_prob:0.5}))
    print('Training accuracy without dropout = ', net.accuracy.eval(feed_dict={net.x:trainIms, net.y_:trainLabs, net.keep_prob:1.0}))

    # Performance on test data
    Inds = np.random.permutation(net.numTestImgs)
    testIms, testLabs = net.createBatch(Inds,'TestData')

    print('Testing accuracy = ', net.accuracy.eval(feed_dict={net.x:testIms, net.y_:testLabs, net.keep_prob:1.0}))












