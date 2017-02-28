

import os
# Path to the weights. Adjust if necessary
path = os.path.expanduser('~/documents/FacePeeper/GenderClassifier/NoAugmentation/')
#path = './GenderClassifier/NoAugmentation/'



def genderization(img):
	'''
	This function receives a face-cropped image of size [112, 112, 3], puts it in our pretrained
	gender classification network and returns the guess of the network in combination with
	the certainty.
	'''

	if img.shape != (112, 112, 3):

		raise ValueError('Please only insert images of size 112x112x3 !')

	import tensorflow as tf
	from tensorflow.core.protobuf import saver_pb2

	# This file lies in the same folder like our network
	from residualCNN import RESNET

	# Initialize network
	net = RESNET(task='GENDER',direc='same')
	net.network()

	# Network expects a batch.
	img = img.reshape([1,112,112,3])

	with tf.Session() as session:

		# Restore the weights
	    saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
	    saver.restore(session, path+"weightsNoAug.ckpt")

	    result = net.OUT.eval(feed_dict={net.x:img, net.keep_prob:1.0})
	    print(result)

	return result





'''from scipy import misc
imP = os.path.expanduser('~/documents/FacePeeper/TestData/0_147.jpg')
img = misc.imread(imP)
a = genderization(img)'''

