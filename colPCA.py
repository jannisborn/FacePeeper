import numpy as np
import glob
from scipy import misc
import os
import time

# Read in Images
numImgs = 5
imgs = np.empty([numImgs,112,112,3])
filenames = os.listdir("C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/")

for ind, filename in enumerate(glob.glob('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/*.jpg')):
#for ind, filename in enumerate(glob.glob('/Users/jannis/Dropbox/github/FacePeeper/Data/*.jpg')):
	if ind < numImgs:
		imgs[ind,:,:,:] = misc.imread(filename) 
print('Images stored')



start = time.time()

# Demeaning
avgIm = np.empty([112,112,3])
for r in range(112):
	for c in range(112):
		for ch in range(3):
			avgIm[r,c,ch] = np.mean(imgs[:,r,c,ch])
imgs -= avgIm
print(time.time()-start)
print('Demeaning done')
start = time.time()


# Reduce Variance
varIm = np.empty([112,112,3])
for r in range(112):
	for c in range(112):
		for ch in range(3):
			varIm[r,c,ch] = np.std(imgs[:,r,c,ch])
imgs /= varIm

print(time.time()-start)
print('Normalization done')

im1 = imgs[1,:,:,:]
misc.imsave('aaa.jpg',im1)
im2 = misc.imread('aaa.jpg')
print(np.amax(im1),np.amin(im1),np.amax(im2),np.amin(im2))
print(im1[7,7,:],im2[7,7,:])
print(im2.dtype)
im3 = im2.astype(np.float64)

im3 *= 4/255
print(np.amax(im3),np.amin(im3))
im3 -= 2
print(np.amax(im3),np.amin(im3))

print(im1[60:62,89,:],im3[60:62,89,:])


print((im1==im2).all())

def compute_PCA(images):
	# Reshape Images to make each pixel a single datapoint in the RGB space
	imgsResh = images.reshape(numImgs*112*112,3)

	# Compute covariance matrix, eigenvectors and eigenvalues
	cov = np.dot(imgsResh.T,imgsResh) / numImgs
	U, S, V = np.linalg.svd(cov)
	eigenvalues = np.sqrt(S)
	print(eigenvalues)
	np.savetxt('eigenvalues.txt',eigenvalues)
	np.savetxt('eigenvectors.txt', U)

def splitTrainTest(images):
	''' This function splits the normalized images into training and testing dataset in a 85/15 ratio. '''
	if not os.path.exists('TrainData'):
		os.makedirs('TrainData')
	if not os.path.exists('TestData'):
		os.makedirs('TestData')

	shuffler = np.random.choice([0,1],images.shape[0],p = [0.85,0.15]) # Randomize for each image whether it is train(0) or test(1)
	counts = np.unique(shuffler,return_counts = True)
	trainData = np.empty([counts[1][0],112,112,3])
	testData = np.empty([counts[1][1],112,112,3])
	countTrain = 0
	countTest = 0

	shuffler = shuffler.astype(bool)
	for ind,img in enumerate(images):
		if shuffler[ind]:
			testData[countTest,:,:,:] = img
			#misc.imsave('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/TestData/'+filenames[ind],img)
		else:
			trainData[countTrain,:,:,:] = img
			#misc.imsave('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/TrainData/'+filenames[ind],img)

	np.savetxt('TrainData.txt',trainData)
	np.savetxt('TestData.txt',testData)





#compute_PCA(imgs)
#splitTrainTest(imgs)

