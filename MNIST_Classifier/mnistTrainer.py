# Test MNIST task
# We used this file to test our network on the MNIST task

# Switch this off if you don't want the images to be augmented before feeded into the net
augmentation = True

# Execute this file while being in the MNIST directory (not from parent directory e.g.)

# Import modules
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Import the network class from .py file in parent directory
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from residualCNN import RESNET


# Initialize net
net = RESNET(task='MNIST',direc='below')
net.network()



# Hyperparameter
epochs = 100
batchSize = 50
batchesPerEpoch = mnist.train.images.shape[0] // batchSize

# Learning Rate Parameter. Plateau detecting, decaying LR
learningRate = 0.01
threshold = 0.01
nextTest = 10
step = 10



# Memory Allocation
ACCS = np.empty([epochs,2]) # First column: TrainData, second: ValData

# Read in Dataset. In this case we store the entire dataset, no loading on the gly
valLabels = mnist.validation.labels
valImgs = np.empty([5000,28,28,3])
for k in range(3):
    valImgs[:,:,:,k] = mnist.validation.images.reshape([5000,28,28])

testLabels = mnist.test.labels
testImgs = np.empty([10000,28,28,3])
for i in range(3):
    testImgs[:,:,:,i] = mnist.test.images.reshape([10000,28,28])


with tf.Session() as session:

	session.run(tf.global_variables_initializer())

	for epoch in range(epochs):

		p = np.empty(batchesPerEpoch)

		for batchN in range(batchesPerEpoch):

			t, trainingLabels = mnist.train.next_batch(batchSize)

			# Our network expects RGB rather than greyscale images, so we extend it to 3 channels
			trainImgs = np.empty([batchSize,28,28,3])
			for k in range(3):
				trainImgs[:,:,:,k] = t.reshape([batchSize,28,28])

			trainImgs = net.augment(trainImgs) if augmentation else trainImgs

			_, p[batchN] = session.run([net.train_step, net.accuracy],
				feed_dict={net.x: trainImgs, net.y_: trainingLabels,net.keep_prob: 0.5, net.lr:learningRate})

		ACCS[epoch,0] = np.mean(p)
		print('Training acuracy after epoch: ', epoch+1, ' = ' , ACCS[epoch,0])

		# Check Validation Performance
		ACCS[epoch,1] = net.accuracy.eval(feed_dict={net.x:valImgs, net.y_:valLabels, net.keep_prob:1.0})
		print('Validation acuracy after epoch: ', epoch+1, ' = ' , ACCS[epoch,1])


		# Feedback based decay of LR, according to performance on validation dataset
		# Divide LR by 5 whenever there was no improvement in the last 5 epochs compared to the 5 before
		if epoch > nextTest and np.mean(ACCS[epoch-(step//2):epoch,1]) < np.mean(ACCS[epoch-step:epoch-(step//2),1]) + threshold:
			
			learningRate /= 5
			threshold /= 6
			nextTest = epoch + 10
			print('New Learning Rate = ', learningRate)

	saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
	saver.save(session, "./weights.ckpt",global_step=epoch)



# Check Test Performance		 
print('Testing accuracy = ', net.accuracy.eval(feed_dict = {net.x: testImgs, net.y_: testLabels, net.keep_prob:1.0}))

