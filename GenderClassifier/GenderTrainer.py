

# Train Gender task
# This file we used to train our network on the gender task

import tensorflow as tf
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

netName = 'GenderClassifier/WithAugmentation/'
if not os.path.exists('./'+netName):
    os.makedirs('./'+netName)

# Hyperparameter
epochs = 1000
batchSize = 50
batchesPerEpoch = gen.numTrainImgs // batchSize


# Learning Rate Parameter
learningRate = 0.01
threshold = 0.005
nextTest = 25
step = 25

# Memory Allocation
accur = np.zeros([epochs,batchesPerEpoch]) # batch-wise
ACCS = np.empty([epochs,2]) # First column: TrainData, second: TestData

with tf.Session() as session:
    
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./GenderClasser/weights.ckpt-10")
    #session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables(), write_version = saver_pb2.SaverDef.V1)
    for epoch in range(11,epochs):

        epochInds = np.random.permutation(gen.numTrainImgs)

        for batchN in range(batchesPerEpoch):

            batchInds = epochInds[(batchN*batchSize):(batchN+1)*batchSize]
            trainIms, trainLabs = gen.createBatch(batchInds, 'trainData')

            trainIms = gen.augment(trainIms)

            _,accur[epoch,batchN] = session.run([train_step,accuracy],feed_dict={x: trainIms, 
                y_: trainLabs, keep_prob: 0.5, lr:learningRate})
            


        ACCS[epoch,0] = np.mean(accur[epoch,:])
        print('Training acuracy after epoch: ', epoch+1, ' = ' , ACCS[epoch,0])


        # Check Test Performance
        inds = np.random.permutation(gen.numTestImgs)
        testIms, testLabs = gen.createBatch(inds,'testData')
        ACCS[epoch,1] = accuracy.eval(feed_dict = {x: testIms, y_: testLabs,keep_prob:1.0})
        print('Testing accuracy after epoch ',epoch+1, ' = ', ACCS[epoch,1])



        # Feedback based decay of LR
        # Divide LR by 10 whenever there was not improvement 

        if epoch > nextTest and learningRate > 1e-8 and np.mean(ACCS[epoch-(step//2):epoch,
            1]) < np.mean(ACCS[epoch-step:epoch-(step//2),1]) + threshold:

            learningRate /= 2
            threshold /= 6
            nextTest = epoch + 25
            print('New Learning Rate = ', learningRate)
   

        if epoch>0 and epoch % 100 == 0:
            saver.save(session, "./"+netName+"/weights.ckpt",global_step=epoch)
            np.savetxt('./'+netName+'/Accuracy.txt',ACCS)
            print('Weights and Accuracies saved')

    saver.save(session, "./"+netName+"/weights.ckpt",global_step=epoch)
    np.savetxt('./'+netName+'/AccuracyFinal.txt',ACCS)
    print('DONE')