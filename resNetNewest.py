import os
import tensorflow as tf
import numpy as np
from scipy import misc
import random
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#dir_train = r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow\TrainData/"
dir_train = "TrainData/"
#dir_test = r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow\TestData/"
dir_test = "TestData/"


class CELEBRITIES():
    def __init__(self):
        self.trainFiles = os.listdir(dir_train)
        self.numTrainImgs = len(self.trainFiles)
        self.testFiles = os.listdir(dir_test)
        self.numTestImgs = len(self.testFiles)
        

    def createBatch(self, indices, batchType = ''):
        '''
        This function receives the indices pointing to the list that contains the names of all
            training/testing images. It loads these images and thus creates the next batch
        Images and labels are returned as np.arrays
        '''
        
       batchSize = len(indices)

       # Define dataset (train/test), error handling
        if batchType == 'trainData':
            path = dir_train
            files = self.trainFiles
        elif batchType == 'testData':
            path = dir_test
            files = self.testFiles
        else:
            raise ValueError("Please call createBatch function with 'trainData' or 'testData' as batchType")

        # Allocate space (append is slow...)
        images = np.empty([batchSize,112,112,3])
        labels = np.empty(batchSize,dtype=int)

        # Load images and labels one by one
        for counter,imgInd in enumerate(indices):
            images[counter,:,:,:] = misc.imread(path + files[imgInd])
            labels[counter] = int(files[imgInd][:3])

        # The images have been normalized before saving. However misc.imsave scales everything to [0,255]
        # The range of zscores after normalization was suprisingly close to [-2,2]. We therefore rescale [0,255] --> [-2,2]
        images = (4*images)/255
        images -= 2
        return images, labels


celeb = CELEBRITIES()


EPSILON = 1e-3

def augment(batch,mu=0,sigma=0.1):
        '''
        This augmentation function is inspired by: Krizhevsky et al. (2012): ImageNet 
            Classification with Deep Convolutional Neural Networks
        This function receives a batch of images (np.array of size [BatchSize,112,112,3]) 
        and performs the following augmentation steps (on the fly, for each img):
            1. Color Augmentation based on channel-wise PCA of entire dataset.
                VarianceIncreaser + NoiseAdder
            2. Binarizes about flipping vertically
            3. Rotation within range of [-25,25] degree
        The optional parameters mu and sigma define the average and the spread in the noise added
        to each RGB channel  '''
        

        batchSize = batch.shape[0]

        # Allocate array for augmented images 
        batchP = np.empty([batchSize,112,112,3],dtype=np.uint8)

        # Restore PCA results that has been performed beforehand on entire dataset
        eigenvalues = np.loadtxt('eigenvalues.txt')
        eigenvectors = np.loadtxt('eigenvectors.txt')
        
        # generate stochastic noise (alpha samples)
        samples = np.random.normal(mu,sigma,[batchSize,3])
        augmentation = samples * eigenvalues # scale by eigenvalue
        
        # augment every image
        for ind,img in enumerate(batch):
            # RGB augmentation via PCA (increase variance, tune luminance+color invariance)
            noise = np.dot(eigenvectors,augmentation[ind])
            img = img + noise # Add color perturbation to image

            # Flip horizontally (eventually)
            img = np.fliplr(img) if np.random.randint(2) else img

            # Rotate randomly 
            dg = random.randint(0,15) if random.randint(2) else -random.randint(0,15)
            batchP[ind] = misc.imrotate(img, dg) 
                
        return batchP


def batch_normalization(x):
    mean, var = tf.nn.moments(x, [0, 1, 2])
    normBatch = tf.nn.batch_normalization(x, mean, var, None, None, EPSILON)
    return normBatch


def output_layer(x, targetDim):
    weights = tf.get_variable("weights", [x.get_shape()[1], targetDim], 
                              tf.float32, tf.random_normal_initializer(stddev = 0.02))
    biases = tf.get_variable("biases", [targetDim], tf.float32, tf.constant_initializer(0.0))
    return tf.matmul(x, weights) + biases


def convLayer(x, targetDim, kernelHeight = 3, kernelWidth = 3, strideX = 1, strideY = 1):
    kernels = tf.get_variable("kernels", [kernelHeight, kernelWidth, x.get_shape()[-1], targetDim],
                             tf.float32, tf.random_normal_initializer(stddev = 0.02))
    conv = tf.nn.conv2d(x, kernels, strides = [1, strideX, strideY, 1], padding = "SAME")
    biases = tf.get_variable("biases", [targetDim], initializer = tf.constant_initializer(0.0))
    activation = tf.nn.bias_add(conv, biases)
    
    activation = batch_normalization(activation)
    
    return activation



def residualBlock(x, targetDim, firstBlock = False):
    # Downsampling is performed directly by convolutional layers with a stride of 2.
    with tf.variable_scope('conv1_in_block'):
        if firstBlock:
            activation = convLayer(x, targetDim, 3, 3, 2, 2)
            inputOut = tf.nn.relu(activation)
        else:
            activation = convLayer(x, targetDim, 3, 3, 1, 1)
            inputOut = tf.nn.relu(activation)
        
    with tf.variable_scope('conv2_in_block'):
        activation = convLayer(inputOut, targetDim, 3, 3, 1, 1)
        out = tf.nn.relu(activation)
    with tf.variable_scope('conv3_in_block'):
        activation = convLayer(out, targetDim, 3, 3, 1, 1)
        residual = activation + inputOut
        inputOut = tf.nn.relu(residual)

    return inputOut



#input images are 112x112x3
images = tf.placeholder(tf.float32, [None, 112, 112, 3])
desired = tf.placeholder(tf.int64, [None])
lr = tf.placeholder(tf.float32, None)


# Batch normalize images
images = batch_normalization(images)


# COMPUTATIONAL GRAPH
# First convolution in each block has stride 2 (i.e., downsampling)
# 64 kernels (3x3), 6 layers, i=3
with tf.variable_scope('round1'):
    inputOut = residualBlock(images, 64, firstBlock = True)
for i in range(2):
    with tf.variable_scope('round1_%d' %i):
        inputOut = residualBlock(inputOut, 64, firstBlock = False)

# 128 kernels (3x3), 8 layers, i=4
with tf.variable_scope('round2'):
    inputOut = residualBlock(inputOut, 128, firstBlock = True)
for i in range(3):
    with tf.variable_scope('round2_%d' %i):
        inputOut = residualBlock(inputOut, 128, firstBlock = False)
    
# 256 kernels (3x3), 12 layers, i=6
with tf.variable_scope('round3'):
    inputOut = residualBlock(inputOut, 256, firstBlock = True)
for i in range(5):
    with tf.variable_scope('round3_%d' %i):
        inputOut = residualBlock(inputOut, 256, firstBlock = False)
    
# 512 kernels (3x3), 6 layers, i=3
with tf.variable_scope('round4'):
    inputOut = residualBlock(inputOut, 512, firstBlock = True)
for i in range(2):
    with tf.variable_scope('round4_%d' %i):
        inputOut = residualBlock(inputOut, 512, firstBlock = False)


out = tf.nn.relu(batch_normalization(inputOut))
flat = tf.reshape(out, [-1, 7 * 7 * 512])
logits = output_layer(flat, 388)

crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, desired)
crossEntropy = tf.reduce_mean(crossEntropy)

# optimizer
trainStep = tf.train.AdamOptimizer(lr).minimize(crossEntropy)
# calculate accuracy
accuracy = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), desired)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

epochs = 600000
batchSize = 256
trainingSteps = 100 * epochs

acc_fig, acc_ax = plt.subplots(1,1)
ce_fig, ce_ax = plt.subplots(1,1)


accur = np.zeros(trainingSteps)
crossEntr = np.zeros(trainingSteps)
#validAccur = np.zeros(trainingSteps)
#validCrossEnt = np.zeros(trainingSteps)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    
    saver = tf.train.Saver()
    
    step = 0
    
    for epoch in range(epochs):
        
        print('epoch: ', epoch)
        epochInds = np.random.permutation(celeb.numTrainImgs)        

        for batchNumber in range(celeb.numTrainImgs//batchSize):
            print('batch_nr: ', batchNumber)

            batchInds = epochInds[(batchNumber*batchSize):(batchNumber+1)*batchSize]

            trainingImages, trainingLabels = celeb.createBatch(batchInds, 'trainData')
            # Data augmentation
            trainingImages = augment(trainingImages)

            stLR = 0.0000001
            #learningRate = tf.train.exponential_decay(starterLearningRate, globalStep, 10e3, 0.96)

            #crossEntr[step], accur[step], _ = session.run([crossEntropy, accuracy, trainStep],
            #                        feed_dict = {images: trainingImages, desired: trainingLabels, lr: stLR})

            #print('Accuracy: ', accur[step])
            prediction = tf.argmax(tf.nn.softmax(logits), 1)
            des = desired
            best, des = session.run([prediction, des], feed_dict = {images: trainingImages, desired: trainingLabels, lr: stLR})
            print('Best prediction: ', best, ' - Desired: ', des)

            if (step % 25 == 0 and step != 0) or step == trainingSteps-1:
                saver.save(session, "./resnet.chkp", step)
                
            step += 1

            #f = plt.figure()
            #x1 = np.linspace(25,epochs,epochs-25)
            #plt.plot(x1,accur[25:])
            #f.savefig('CELEBRITIES.png')
 
with tf.Session() as session:
    saver = tf.train.Saver()
    # We restore the weights saved in training and test them on test data.
    saver.restore(session, "./resnet.chkp-" + str(trainingSteps-1))    
    
    accuracies = []
    for r in range(self.numTestImgs//batchSize):

        batchInds = testInds[(r*batchSize):(r+1)*batchSize]
        testImages, testLabels = celeb.createBatch(batchInds, 'testData')
        crossEntr, accur = session.run([crossEntropy, accuracy], feed_dict = {images: testImages, desired: testLabels})
        accuracies.append(accur)

    print(np.mean(accuracies))


def predict_identity(img):
    '''
    This function receives an image that has been initially uploaded by the user on the website.
        This image was face cropped by GUI_prep.py and subsequently confirmed by the user as the 
        face he wants the algorithm to process.

        It returns the five most likely labels for the inserted image
    '''

def retrain(img,label):
    '''
    In case the first guess of the network was wrong and the user returned the true label, 
        this function can be used to retrain the network on the uploaded image with the 
        correct label.
    '''


