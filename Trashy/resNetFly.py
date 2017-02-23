import os
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import preProcess


class CELEBRITIES():
    def __init__(self, directory=r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow"):
    	self.filenames = os.listdir(directory)
		self.numImgs = len(self.filenames)


	def createBatch(batchSize, batchType):
	'''
	This function receives a batchSize and randomly loads some images from the dataset.
	Images and labels are returned as np.array
	'''
		if batchType == 'trainData':
			directory = r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow\trainData"
		elif batchType == 'testData':
			directory = r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow\testData"


		# Variant to read via scipy to np array (maybe slower than TF build in)
		indices = np.random.randint(0,numImgs,batchSize)
		images = np.empty([batchSize,112,112,3])
		labels = np.empty(batchSize,dtype=int)

		for counter,imgInd in enumerate(indices):
			print(counter)
			images[counter,:,:,:] = misc.imread(directory+filenames[self.imgInd])
			labels[counter] = int(self.filenames[self.imgInd][:3])

        # The images have been normalized before saving. However misc.imsave scales everything to [0,255]
        # The range of zscores after normalization was suprisingly close to [-2,2]. We therefore rescale [0,255] --> [-2,2]
        images = (4*images)/255
        images -= 2
		return images,labels


	'''
    # Shuffle the samples and to pack them into equally sized batches.
    def shuffleSamples(self, batchSize = 50):
        indices = np.random.permutation(len(self.trainingData))
        self.data = self.trainingData[indices]
        self.labels = self.trainingLabels[indices]
        
        self.dataBatches = []
        self.labelBatches = []
        
        for i in range(len(self.data) // batchSize):
            ib = i * batchSize
            self.dataBatches.append(np.array(self.data[ib:ib+batchSize]))
            self.labelBatches.append(np.array(self.labels[ib:ib+batchSize]))
    
    # Retrieve the training, validation and test data
    def getTrainingData(self):
        return self.dataBatches, self.labelBatches

    def getValidationData(self):
        return self.validationData, self.validationLabels
    
    def getTestData(self):
        return self.testData, self.testLabels
	'''


celeb = CELEBRITIES("./")



EPSILON = 1e-3

def batch_normalization(x):
    mean, var = tf.nn.moments(x, [0, 1, 2])
    normBatch = tf.nn.batch_normalization(x, mean, var, None, None, EPSILON)
    return normBatch


def output_layer(x, targetDim):
    weights = tf.get_variable("weights", [x.get_shape()[1], targetDim], 
                              tf.float32, tf.random_normal_initializer(stddev = 0.02))
    biases = tf.get_variable("biases", [targetDim], tf.float32, tf.constant_initializer(0.0))
    return tf.matmul(x, weights) + biases


def convLayer(x, filter_shape, stride):
    '''
    Batch normalization, relu and 2D-convolution
    :param x: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for convolution
    :return: 4D tensor.
    '''
    kernels = tf.get_variable("kernels", filter_shape, tf.float32, tf.random_normal_initializer(stddev = 0.02))
    
    normBatch = batch_normalization(x)
    reluStep = tf.nn.relu(normBatch)
    
    conv = tf.nn.conv2d(reluStep, kernels, strides=[1, stride, stride, 1], padding="SAME")
    
    return conv


def residual_block(x, output_channel):
    '''
    Defines a residual block in ResNet
    :param x: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :return: 4D tensor.
    '''
    input_channel = x.get_shape().as_list()[-1]

    # When it's time to halve the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channels do not match in residual blocks!')

    # 1st convolution in block can have stride 1 or 2
    with tf.variable_scope('conv1_in_block'):
        conv1 = convLayer(x, [3, 3, input_channel, output_channel], stride)
    
    # 2nd convolution in block has stride 1
    with tf.variable_scope('conv2_in_block'):
        conv2 = convLayer(conv1, [3, 3, output_channel, output_channel], 1)

    # When size of x and conv2 does not match, we add zero pads to increase the
    #  depth of x's
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = x

    output = conv2 + padded_input
    
    return output

#input images are 112x112x3
images = tf.placeholder(tf.float32, [None, 112, 112, 3])
desired = tf.placeholder(tf.int64, [None])
lr = tf.placeholder(tf.float32, None)


# Batch normalize images
images = batch_normalization(images)

# ResNet. Total layers: 6 + 8 + 12 + 6 + 1 (33 layers)
layers = []

# 64 kernels (3x3), 6 layers, i=3
for i in range(3):
    with tf.variable_scope('conv1_%d' %i):
        if i==0:
            conv1 = residual_block(images, 64)
        else:
            conv1 = residual_block(layers, 64)
        layers.append(conv1)

# 128 kernels (3x3), 8 layers, i=4
for i in range(4):
    with tf.variable_scope('conv2_%d' %i):
        conv2 = residual_block(layers, 128)
        layers.append(conv2)
    
# 256 kernels (3x3), 12 layers, i=6
for i in range(6):
    with tf.variable_scope('conv3_%d' %i):
        conv3 = residual_block(layers, 256)
        layers.append(conv3)
    
# 512 kernels (3x3), 6 layers, i=3
for i in range(3):
    with tf.variable_scope('conv4_%d' %i):
        conv4 = residual_block(layers, 512)
        layers.append(conv4)


# Batch normalize layers, relu, global_avg_pool
layers = batch_normalization(layers)
reluLayer = tf.nn.relu(layers)
globalPool = tf.reduce_mean(reluLayer, [1,2])
output = output_layer(globalPool, 400)
layers.append(output)

logits = layers[-1]

crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, desired)
crossEntropy = tf.reduce_mean(crossEntropy)

# Passing global_step to minimize() will increment it at each step.
trainStep = tf.train.AdamOptimizer(lr).minimize(crossEntropy, globalStep=globalStep)
#trainStep = tf.train.AdamOptimizer(lr).minimize(crossEntropy)

accuracy = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), desired)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

epochs = 60*10e3
batchSize = 256
#celeb.shuffleSamples(batchSize)
#trainingSteps = len(celeb.getTrainingData()[0]) * epochs
trainingSteps = 50000 * epochs

acc_fig, acc_ax = plt.subplots(1,1)
ce_fig, ce_ax = plt.subplots(1,1)

accur = np.zeros(trainingSteps)
crossEntr = np.zeros(trainingSteps)
validAccur = np.zeros(trainingSteps)
validCrossEnt = np.zeros(trainingSteps)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    step = 0
    
    for epoch in range(epochs):
        imageBatches, labelBatches = createBatch(batchSize, 'trainData')
        
        # Data augmentation
        imageBatches = preProcess(imageBatches)
        
        for batchNumber in range(len(imageBatches)):
            trainingImages = imageBatches[batchNumber]
            trainingLabels = labelBatches[batchNumber]
            
            globalStep = tf.Variable(0, trainable=False)
            starterLearningRate = 0.1
            learningRate = tf.train.exponential_decay(starterLearningRate, globalStep, 10e3, 0.96, staircase)

            crossEntr[step], accur[step], _ = session.run([crossEntropy, accuracy, trainStep],
                                      feed_dict = {images: trainingImages, desired: trainingLabels, lr: learningRate})

            if (step % 500 == 0 and step != 0) or step == trainingSteps-1:
                saver.save(session, "./resnet.chkp", step)

            step += 1
 
with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session, "./resnet.chkp-" + str(trainingSteps-1))
    
    testImages, testLabels = celeb.createBatch(batchSize, 'testData')
    accuracies = []
    for r in range(0, len(testImages), 1000):
        crossEntr, accur = session.run([crossEntropy, accuracy], feed_dict = {images: testImages[r:r+1000], desired: testLabels[r:r+1000]})
        accuracies.append(accur)
    print(np.mean(accuracies))