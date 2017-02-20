# Ideas. No guarantee for work

# Add this to the import section on top
import os
import numpy as np
from scipy import misc

# Add this after the tf.placeholder:
self.filenames = os.listdir("./Data/")
self.numImgs = len(self.filenames)



# Call this function when you want to get a batch
def createBatch(batchSize):
	''' This function receives a batchSize and randomly loads some images from the dataset.
	Images and labels are returned as np.array '''

	# Variant to read via scipy to np array (maybe slower than TF build in)
	indices = np.random.randint(0,numImgs,batchSize)

	images = np.empty([batchSize,112,112,3])
	labels = np.empty(batchSize,dtype=int)

	for counter,imgInd in enumerate(indices):
		print(counter)
		images[counter,:,:,:] = misc.imread('./Data/'+filenames[self.imgInd])
		labels[counter] = int(self.filenames[self.imgInd][:3])
	return images,labels

	
# imgs, labels = createBatch(20)
