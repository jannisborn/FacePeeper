# Ideas. No guarantee for work


# Add this after the tf.placeholder:
self.filenames = tf.train.match_filenames_once("./images/*.jpg")
self.numImgs = len(filename_queue)



# Call this function when you want to get a batch
def createBatch(batchSize):
	''' This function receives a batchSize and randomly loads some images from the dataset.
	Images and labels are returned as np.array '''

	from scipy import misc

	# Variant to read via scipy to np array (maybe slower than TF build in)
	indices = np.random.randint(0,self.numImgs,batchSize)
	images = np.empty([batchSize,112,112,3])
	labels = np.array(batchSize,dtype=int)
	for counter,imgInd in enumerate(indices):
		images[counter,:,:,:] = misc.imread(self.filenames[imgInd])
		labels[counter] = int(self.filenames[imgInd][:3])
	return images,labels

	
