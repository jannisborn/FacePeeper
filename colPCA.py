import numpy as np
import glob
from scipy import misc

# Read in Images
numImgs = 50312
imgs = np.empty([numImgs,112,112,3])
for ind, filename in enumerate(glob.glob('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/*.jpg')):
#for ind, filename in enumerate(glob.glob('/Users/jannis/Dropbox/github/FacePeeper/Data/*.jpg')):
	img = misc.imread(filename) 
	img2 = img * (1/np.amax(img)) # Normalize all images
	imgs[ind,:,:,:] = img2
print('Images stored')

def compute_PCA(images):
	# Reshape Images to make each pixel a single datapoint in the RGB space
	imgsResh = images.reshape(numImgs*112*112,3)

	# Compute covariance matrix, eigenvectors and eigenvalues
	cov = np.dot(imgsResh.T,imgsResh) / numImgs
	U, S, V = np.linalg.svd(cov)
	eigenvalues = np.sqrt(S)

	np.savetxt('eigenvalues.txt',eigenvalues)
	np.savetxt('eigenvectors.txt', U)

compute_PCA(imgs)

