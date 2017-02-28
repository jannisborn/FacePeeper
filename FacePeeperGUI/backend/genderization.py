
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

# This file lies in the same folder like our network
from backend.residualCNN import RESNET

import os
# Path to the weights. Adjust if necessary
path = os.path.expanduser('/home/ubuntu/documents/FacePeeper/FacePeeperGUI/backend/')
#path = './GenderClassifier/NoAugmentation/'

global net
net = RESNET(task='GENDER',direc=None)
net.network()

def genderization(img, lagel=None):
	'''
	This function provides the interface between the webserver and the network.
	It is called from the webserver for two different usecases:

		1. Called with 'genderization(img)'. Then it expects a face-cropped image of size 
			[112, 112, 3], puts it in our pretrained gender classification network and returns 
			the guess of the network in combination with the certainty (outcome of softmax)

		2. Called with 'genderization(img, label)'. Then it additionally expects a label for 
			the image in format (2,). It then retrains our pretrained network with this label.
			The label is provided from an user of the website. Finally this function returns the 
			new prediction of the network in the same format like in 1.).
			Thus the user can verify the effect of training the network on that picture.		
	'''


	# Error handling
	if img.shape != (112, 112, 3):

		raise ValueError('Please only insert images of size 112x112x3 !')


	# Network expects a batch.
	img = img.reshape([1,112,112,3])

	with tf.Session() as session:

		session.run(tf.global_variables_initializer())

		# Restore the weights
		saver = tf.train.Saver(tf.trainable_variables(),write_version = saver_pb2.SaverDef.V1)
		saver.restore(session, path+"weightsGender.ckpt")

		# Distinguish the two usecases 
		if label == None:
			
			result = net.OUT.eval(feed_dict={net.x:img, net.keep_prob:1.0})

		else:

			session.run(net.train_step,feed_dict={net.x: img, net.y_: label.reshape([1,2]) ,net.keep_prob: 1.0, net.lr:0.004})
			result = net.OUT.eval(feed_dict={net.x:img, net.keep_prob:1.0})
			saver.save(session, path+"weightsGenderWithAug.ckpt")



	tf.reset_default_graph()
	return result

