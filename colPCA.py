import numpy as np
import glob
from scipy import misc
import os

# Read in Images
numImgs = 50312
imgs = np.empty([numImgs,112,112,3])
filenames = os.listdir("C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/")

for ind, filename in enumerate(glob.glob('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/*.jpg')):
#for ind, filename in enumerate(glob.glob('/Users/jannis/Dropbox/github/FacePeeper/Data/*.jpg')):
	imgs[ind,:,:,:] = misc.imread(filename) 
print('Images stored')

# Normalize all images
imgs -= np.mean(imgs, axis = 0)
print('Demeaning done')
imgs /= np.std(imgs, axis = 0)

print('Normalization done')

def compute_PCA(images):
	# Reshape Images to make each pixel a single datapoint in the RGB space
	imgsResh = images.reshape(numImgs*112*112,3)

	# Compute covariance matrix, eigenvectors and eigenvalues
	cov = np.dot(imgsResh.T,imgsResh) / numImgs
	U, S, V = np.linalg.svd(cov)
	eigenvalues = np.sqrt(S)

	np.savetxt('eigenvalues.txt',eigenvalues)
	np.savetxt('eigenvectors.txt', U)

def splitTrainTest(images):
	''' This function splits the normalized images into training and testing dataset in a 85/15 ratio. '''
	os.makedirs('TrainData')
	os.makedirs('TestData')
	shuffler = np.random.choice([0,1],images.shape[0],p = [0.85,0.15]) # Randomize for each image whether it is train(0) or test(1)
	
	for img,ind in enumerate(imgs):
		if shuffler[ind]:
			misc.imsave('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/TrainData/'+filenames[ind],img)
		else:
			misc.imsave('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/TestData/'+filenames[ind],img)








compute_PCA(imgs)
splitTrainTest(imgs)

