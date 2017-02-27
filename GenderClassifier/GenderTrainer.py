
# Train Gender task
# This file we used to train our network on the gender task

# Switch this off if you don't want the images to be augmented before feeded into the net
augmentation = True

# Execute this file while being in the GenderClassifier directory (not from parent directory e.g.)

# Import modules
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np

# Import the network class from .py file in parent directory
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residualCNN import RESNET

# Initialize net
net = RESNET(task='GENDER',direc='below')
net.network()



# Hyperparameter
epochs = 500
batchSize = 50
batchesPerEpoch = net.numTrainImgs // batchSize

# Learning Rate Parameter
# We implement a feedback-based decaying learning that starts with 0.01 and decays whenever learning plateaus
# over the last 25 epochs. A plateau is initially defined as an improvement of less then 0.5% over 25 epochs 
# (hence threshold = 0.005) but decays over time as well.
learningRate = 0.01
threshold = 0.005
nextTest = 25
step = 25


# Memory Allocation
accur = np.zeros([epochs,batchesPerEpoch]) # batch-wise
ACCS = np.empty([epochs,2]) # First column: TrainData, second: TestData

with tf.Session() as session:
    
    saver = tf.train.Saver(tf.trainable_variables(), write_version = saver_pb2.SaverDef.V1)
    session.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        # Randomize order of training data at beginning of every epoch
        epochInds = np.random.permutation(net.numTrainImgs)

        p = np.empty(batchesPerEpoch)

        for batchN in range(batchesPerEpoch):

            batchInds = epochInds[(batchN*batchSize):(batchN+1)*batchSize]
            trainIms, trainLabs = net.createBatch(batchInds, 'TrainData')

            trainIms = net.augment(trainIms) if augmentation else trainIms

            _,p[batchN] = session.run([net.train_step,net.accuracy],feed_dict={net.x: trainIms, 
                net.y_: trainLabs, net.keep_prob: 0.5, net.lr:learningRate})
            

        ACCS[epoch,0] = np.mean(p[batchN])
        print('Training acuracy after epoch: ', epoch+1, ' = ' , ACCS[epoch,0])


        # Check Test Performance
        inds = np.random.permutation(net.numTestImgs)
        testIms, testLabs = net.createBatch(inds,'TestData')
        ACCS[epoch,1] = net.accuracy.eval(feed_dict = {net.x: testIms, net.y_: testLabs, net.keep_prob:1.0})
        print('Testing accuracy after epoch ',epoch+1, ' = ', ACCS[epoch,1])


        # Feedback based decay of LR
        # Divide LR by 10 whenever there was not improvement 
        if epoch > nextTest and learningRate > 1e-8 and np.mean(ACCS[epoch-(step//2):epoch,
            1]) < np.mean(ACCS[epoch-step:epoch-(step//2),1]) + threshold:

            learningRate /= 2
            threshold /= 6
            nextTest = epoch + 25
            print('New Learning Rate = ', learningRate)
   
