

# Test Gender task
# This file we used to test our network on the gender task

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np
import os
import sys

# Specify path to the main directory of the FacePeeper folder
path = '~/documents/FacePeeper/'


# Import network class
sys.path.insert(0,path)
from residualCNN import RESNET

# Specify task and build network
net = RESNET(task='GENDER')
net.network()



### ONLY FOR TESTING, NOT FOR RETRAINING (variables not stored)

with tf.Session() as session:

    saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./NoAugmentation/weights.ckpt")

    Inds = np.random.permutation(net.numTrainImgs)
    trainIms, trainLabs = net.createBatch(Inds, 'trainData')
    print('Training accuracy with dropout = ', net.accuracy.eval(feed_dict={x:trainIms, y_:trainLabs, keep_prob:0.5}))
    print('Training accuracy without dropout = ', net.accuracy.eval(feed_dict={x:trainIms, y_:trainLabs, keep_prob:1.0}))



    Inds = np.random.permutation(gen.numTestImgs)
    testIms, testLabs = net.createBatch(Inds,'testData')

    print('Testing accuracy = ', net.accuracy.eval(feed_dict={x:testIms, y_:testLabs, keep_prob:1.0}))










