# This class implements our 9-layer CNN with 3 residual layers

# Load Packages
import tensorflow as tf
import numpy as np
import os
from scipy import misc
import time
from tensorflow.examples.tutorials.mnist import input_data # for MNIST example
from tensorflow.core.protobuf import saver_pb2 # We prefer this saver structure

class RESNET():


    def __init__(self, task, direc):

        # Weights should be located in the main location from which the testFile is executed that imports this class
        self.weights = "./weights.ckpt"

        # Infer your path
        if direc == 'same':
            self.path = os.getcwd()
        elif direc == 'below':
            self.path = os.path.abspath(os.path.join('./', os.pardir))

        self.task = task


        # Task initialization

        if task == 'MNIST':

            self.classes = 10

            self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            
            self.shape = [28, 28, 3]

           
        elif task == 'IDENTITY':

            self.classes = 10

            # Workaround to load Images on the fly
            trainFiles = os.listdir(self.path+"/TrainData/")
            self.trainFiles = [t for t in trainFiles if t[0] != '.' ]
            self.numTrainImgs = len(self.trainFiles)

            testFiles = os.listdir(self.path+"/TestData/")
            self.testFiles = [t for t in testFiles if t[0] != '.']
            self.numTestImgs = len(self.testFiles)

            self.shape = [112, 112, 3]


        elif task == 'GENDER':

            # Gender classification task
            self.classes = 2

            # Data is labelled according to 10 identites. This are the identity --> gender mappings
            self.men = [1, 3, 5, 7, 9]
            self.women = [0, 2, 4, 6, 8]

            # Workaround to load Images on the fly
            trainFiles = os.listdir(self.path+"/TrainData/")
            self.trainFiles = [t for t in trainFiles if t[0] != '.' ]
            self.numTrainImgs = len(self.trainFiles)

            testFiles = os.listdir(self.path+"/TestData/")
            self.testFiles = [t for t in testFiles if t[0] != '.']
            self.numTestImgs = len(self.testFiles)

            self.shape = [112, 112, 3]

        else:

            raise ValueError("Please insert a proper task, i.e. MNIST, IDENTITY or GENDER. ")


    def createBatch(self, indices, batchType = ''):
        '''
        This function receives the indices pointing to the list that contains the names of all
            training/testing images. It loads these images and thus creates the next batch
            Images and labels are returned as np.arrays.

            Only for IDENTITY and GENDER task needed
        '''

        batchSize = len(indices)

        # Define dataset (train/test), error handling
        if batchType == 'TrainData':
            files = self.trainFiles
        elif batchType == 'TestData':
            files = self.testFiles
        else:
            raise ValueError("Please call createBatch function with 'TrainData' or 'TestData' as batchType")

        # Allocate space (append is slow...)
        images = np.empty([batchSize,112,112,3])
        labels = np.zeros([batchSize,self.classes],dtype=int)

        # Load images and labels one by one
        for counter,imgInd in enumerate(indices):

            images[counter,:,:,:] = misc.imread(self.path + '/' + batchType + '/' + files[imgInd])

            if self.task == 'GENDER':
                labels[counter] = [1,0] if int(files[imgInd][0]) in self.women else [0,1]
            else: # else Identity Task (no other possibility)
                labels[counter,int(files[imgInd][0])] = 1

        # The images have been normalized before saving. However misc.imsave scales everything to [0,255]
        # The range of zscores after normalization was suprisingly close to [-2,2]. 
        # We therefore rescale [0,255] --> [-2,2]

        images = (4*images)/255
        images -= 2

        return images, labels


    def augment(self,batch,mu=0,sigma=0.01):
        '''
        This augmentation function is inspired by: Krizhevsky et al. (2012): ImageNet 
            Classification with Deep Convolutional Neural Networks
        This function receives a batch of images (np.array of size [BatchSize,112,112,3]) 
        and performs the following augmentation steps (on the fly, for each img):
            1. Color Augmentation based on channel-wise PCA of entire dataset.
                VarianceIncreaser + NoiseAdder
            2. Binarizes about flipping vertically
            3. Rotation within range of [-15,15] degree
        The optional parameters mu and sigma define the average and the spread in the noise added
        to each RGB channel  

        Only for IDENTITY and GENDER task needed

        '''
        
        # Restore PCA results that has been performed beforehand on entire dataset
        # For color augmentation

        self.eigenvalues = np.loadtxt(self.path+'/eigenvalues.txt')
        self.eigenvectors = np.loadtxt(self.path+'/eigenvectors.txt')

        batchSize = batch.shape[0]

        # Allocate array for augmented images 
        batchP = np.empty([batchSize,batch.shape[1],batch.shape[2],batch.shape[3]],dtype=np.uint8)

        # generate stochastic noise (alpha samples)
        samples = np.random.normal(mu,sigma,[batchSize,3])
        augmentation = samples * self.eigenvalues # scale by eigenvalue

        # augment every image
        for ind,img in enumerate(batch):

            # RGB augmentation via PCA (increase variance, tune luminance+color invariance)
            img += np.dot(self.eigenvectors,augmentation[ind]) # Add color perturbation to image

            # Flip vertically (eventually)
            img = np.fliplr(img) if np.random.randint(2) else img

            # Rotate randomly 
            dg = np.random.randint(0,15) if np.random.randint(2) else -np.random.randint(0,15)
            img = misc.imrotate(img, dg)
            
            batchP[ind] = img

        return batchP


        # Some functions to build up the network easier

    def weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):

        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(self, x, W, strideX = 1, strideY = 1):

        # Convolution and batch normalization
        conv = tf.nn.conv2d(x, W, strides=[1, strideX, strideY, 1], padding='SAME')
        mean, var = tf.nn.moments(conv, [0, 1, 2])
        activation = tf.nn.batch_normalization(conv, mean, var, None, None, 1e-3)
        return activation


    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    def network(self):

        # This function creates the network

        self.x = tf.placeholder(tf.float32, shape=[None, self.shape[0], self.shape[1], self.shape[2]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.classes])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        # For "global normalization" used with convolutional
        #   filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
        mean, var = tf.nn.moments(self.x, [0, 1, 2])
        image = tf.nn.batch_normalization(self.x, mean, var, None, None, 1e-3)



        # 6 convolutions - 64 kernels (3x3)
        W_conv1 = self.weight_variable([3,3,3,64])
        b_conv1 = self.bias_variable([64])

        # Feature map size is halved by stride of 2
        activation = self.conv2d(image, W_conv1, 2, 2) + b_conv1
        h_conv1 = tf.nn.relu(activation)

        W_conv2 = self.weight_variable([3,3,64,64])
        b_conv2 = self.bias_variable([64])
        activation = self.conv2d(h_conv1, W_conv2, 1, 1) + b_conv2
        h_conv2 = tf.nn.relu(activation)

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])
        activation = self.conv2d(h_conv2, W_conv3, 1, 1) + b_conv3
        residual = activation + h_conv1
        h_conv3 = tf.nn.relu(activation)

        W_conv4 = self.weight_variable([3,3,64,64])
        b_conv4 = self.bias_variable([64])
        activation = self.conv2d(h_conv3, W_conv4, 1, 1) + b_conv4
        h_conv4 = tf.nn.relu(activation)

        W_conv5 = self.weight_variable([3,3,64,64])
        b_conv5 = self.bias_variable([64])
        activation = self.conv2d(h_conv4, W_conv5, 1, 1) + b_conv5
        h_conv5 = tf.nn.relu(activation)

        W_conv6 = self.weight_variable([3,3,64,64])
        b_conv6 = self.bias_variable([64])
        activation = self.conv2d(h_conv5, W_conv6, 1, 1) + b_conv6
        residual = activation + h_conv4
        h_conv6 = tf.nn.relu(activation)

        # 3 convolutions - 128 kernels (3x3)
        W_conv7 = self.weight_variable([3,3,64,128])
        b_conv7 = self.bias_variable([128])
        # Halve size of map features
        activation = self.conv2d(h_conv6, W_conv7, 2, 2) + b_conv7
        h_conv7 = tf.nn.relu(activation)

        W_conv8 = self.weight_variable([3,3,128,128])
        b_conv8 = self.bias_variable([128])
        activation = self.conv2d(h_conv7, W_conv8, 1, 1) + b_conv8
        h_conv8 = tf.nn.relu(activation)

        W_conv9 = self.weight_variable([3,3,128,128])
        b_conv9 = self.bias_variable([128])
        activation = self.conv2d(h_conv8, W_conv9, 1, 1) + b_conv9
        residual = activation + h_conv7
        h_conv9 = tf.nn.relu(activation)

        # Image size has been cutted down to a quarter
        W_fc1 = self.weight_variable([(self.shape[0]//4)*(self.shape[1]//4)*128, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_conv8, [-1, (self.shape[0]//4)*(self.shape[1]//4)*128])


        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([1024, self.classes])
        b_fc2 = self.bias_variable([self.classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.OUT = tf.nn.softmax(y_conv)

        # Readout functions

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.OUT,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))












