import tensorflow as tf
import numpy as np
import os
from scipy import misc
from PIL import Image
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.protobuf import saver_pb2




dir_train = "TrainData/"
dir_test = "TestData/"

netName = 'GenderClassifier/NoAugmentation/'
if not os.path.exists('./'+netName):
    os.makedirs('./'+netName)




class GENDER():
    def __init__(self):

        # Workaround to load Images on the fly
        trainFiles = os.listdir(dir_train)
        self.trainFiles = [t for t in trainFiles if t[0] != '.' ]
        self.numTrainImgs = len(self.trainFiles)

        testFiles = os.listdir(dir_test)
        self.testFiles = [t for t in testFiles if t[0] != '.']
        self.numTestImgs = len(self.testFiles)

        # Gender classification task
        self.classes = 2

        # Data is labelled according to 10 identites. This are the identity --> gender mappings
        self.men = [1, 3, 5, 7, 9]
        self.women = [0, 2, 4, 6, 8]

        # Restore PCA results that has been performed beforehand on entire dataset
        # Needed for augmentation function
        self.eigenvalues = np.loadtxt('eigenvalues.txt')
        self.eigenvectors = np.loadtxt('eigenvectors.txt')


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
        labels = np.zeros([batchSize,self.classes],dtype=int)

        # Load images and labels one by one
        for counter,imgInd in enumerate(indices):
            images[counter,:,:,:] = misc.imread(path + files[imgInd])
            labels[counter] = [1,0] if int(files[imgInd][0]) in self.women else [0,1]
            #labels[counter,int(files[imgInd][0])] = 1

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
        to each RGB channel  '''


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


gen = GENDER()

################## The 9 layer residual CNN  ##################

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, strideX = 1, strideY = 1):
  # Convolution and batch normalization
  conv = tf.nn.conv2d(x, W, strides=[1, strideX, strideY, 1], padding='SAME')
  mean, var = tf.nn.moments(conv, [0, 1, 2])
  activation = tf.nn.batch_normalization(conv, mean, var, None, None, 1e-3)
  return activation

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




x = tf.placeholder(tf.float32, shape=[None, 112, 112, 3])
y_ = tf.placeholder(tf.float32, shape=[None, gen.classes])
lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# For "global normalization" used with convolutional
#   filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
mean, var = tf.nn.moments(x, [0, 1, 2])
image = tf.nn.batch_normalization(x, mean, var, None, None, 1e-3)



# 6 convolutions - 64 kernels (3x3)
W_conv1 = weight_variable([3,3,3,64])
b_conv1 = bias_variable([64])

# Feature map size is halved by stride of 2
activation = conv2d(image, W_conv1, 2, 2) + b_conv1
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


W_fc1 = weight_variable([28*28*128, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_conv8, [-1, 28*28*128])


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, gen.classes])
b_fc2 = bias_variable([gen.classes])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
OUT = tf.nn.softmax(y_conv)

################### END NETWORK ####################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Hyperparameter
epochs = 11
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

'''with tf.Session() as session:
    
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./GenderClassifier/NoAugmentation/weights.ckpt-900")
    #session.run(tf.global_variables_initializer())
    #saver = tf.train.Saver(tf.trainable_variables(), write_version = saver_pb2.SaverDef.V1)
    for epoch in range(0,epochs):

        epochInds = np.random.permutation(gen.numTrainImgs)

        for batchN in range(batchesPerEpoch):

            batchInds = epochInds[(batchN*batchSize):(batchN+1)*batchSize]
            trainIms, trainLabs = gen.createBatch(batchInds, 'trainData')

            #trainIms = gen.augment(trainIms)

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
    np.savetxt('./'+netName+'/AccuracyFirst10.txt',ACCS)
    print('DONE')'''

### ONLY FOR TESTING, NOT FOR RETRAINING (variables not stored)

with tf.Session() as session:

    saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
    saver.restore(session, "./GenderClassifier/WithAugmentation/weights.ckpt-500")

    Inds = np.random.permutation(gen.numTrainImgs)
    trainIms, trainLabs = gen.createBatch(Inds, 'trainData')
    print('Training accuracy = ', accuracy.eval(feed_dict={x:trainIms, y_:trainLabs, keep_prob:0.5}))
    print('Training accuracy = ', accuracy.eval(feed_dict={x:trainIms, y_:trainLabs, keep_prob:1.0}))



    Inds = np.random.permutation(gen.numTestImgs)
    testIms, testLabs = gen.createBatch(Inds,'testData')

    print('Testing accuracy = ', accuracy.eval(feed_dict={x:testIms, y_:testLabs, keep_prob:1.0}))











