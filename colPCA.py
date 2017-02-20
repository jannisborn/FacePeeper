import numpy as np
import glob
from scipy import misc

# Read in Images
numImgs = 50312
imgs = np.empty([numImgs,112,112,3])
for ind, filename in enumerate(glob.glob('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/*.jpg')):
#for ind, filename in enumerate(glob.glob('/Users/jannis/Dropbox/github/FacePeeper/Data/*.jpg')):
	imgs[ind,:,:,:] = misc.imread(filename) 
print('Images stored')

def compute_PCA(images):
	# Reshape Images to make each pixel a single datapoint in the RGB space
	imgsResh = images.reshape(numImgs*112*112,3)

	# Compute covariance matrix, eigenvectors and eigenvalues
	cov = np.dot(imgsResh.T,imgsResh) / numImgs
	U, S, V = np.linalg.svd(cov)
	eigenvalues = np.sqrt(S)

	np.savetxt('eigenvalues',eigenvalues)
	np.savetxt('eigenvectors', U)

compute_PCA(imgs)

