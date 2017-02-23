# Ideas. No guarantee for work

# Add this to the import section on top
import os
import numpy as np
from scipy import misc
import time

# Add this after the tf.placeholder:
filenames = os.listdir("C:/Users/Jannis/Dropbox/Github/FacePeeper/TrainData/")
numImgs = len(filenames)

s = time.time()

# Call this function when you want to get a batch
def createBatch(batchSize):
	''' This function receives a batchSize and randomly loads some images from the dataset.
	Images and labels are returned as np.array '''

	# Variant to read via scipy to np array (maybe slower than TF build in)
	indices = np.arange(1,1+batchSize)

	images = np.empty([batchSize,112,112,3])
	labels = np.empty(batchSize,dtype=int)

	for counter,imgInd in enumerate(indices):
		images[counter,:,:,:] = misc.imread('C:/Users/Jannis/Dropbox/Github/FacePeeper/TrainData/'+filenames[imgInd])
		labels[counter] = int(filenames[imgInd][:3])

	# The images have been normalized before saving. However misc.imsave scales everything to [0,255]
	# The range of zscores after normalization was suprisingly close to [-2,2]. We therefore rescale [0,255] --> [-2,2]
	images = (4*images)/255
	images -= 2
	
	return images,labels

	
imgs, labels = createBatch(50)



def preProcess(batch,mu=0,sigma=0.01):

    '''
    This preprocessing function is inspired by: Krizhevsky et al. (2012): ImageNet 
        Classification with Deep Convolutional Neural Networks

    This function receives a batch of images (np.array of size [BatchSize,112,112,3]) 
    and performs the following preprocessing steps (for each img):
        1. Color Augmentation based on channel-wise PCA of entire dataset.
            VarianceIncreaser + NoiseAdder
        2. Binarizes about flipping vertically
        3. Rotation within range of [-25,25] degree
        
    Afterwards it returns a TF Tensor

    The optional parameters mu and sigma define the average and the spread in the noise added
    to each RGB channel 
    '''
    import random
    from PIL import Image
    from scipy import misc
    import numpy as np

    batchSize = batch.shape[0]
    batchP = np.empty([batchSize,112,112,3],dtype=np.uint8)
    # Restore PCA results that has been performed beforehand on entire dataset
    eigenvalues = np.loadtxt('eigenvalues.txt')
    eigenvectors = np.loadtxt('eigenvectors.txt')
    # generate Alpha Samples (to add noise)
    samples = np.random.normal(mu,sigma,[batchSize,3])
    augmentation = samples * eigenvalues # scale by eigenvalue
    for ind,img in enumerate(batch):

        # RGB augmentation via PCA (increase variance, tune luminance+color invariance)
        noise = np.dot(eigenvectors,augmentation[ind])
        img = img + noise # Add color perturbation to image



        #Flip Veritcally?
        #img = tf.random_flip_left_right(tf.convert_to_tensor(img))
        #img = tf.convert_to_tensor(img)

        # Rotate randomly 
        dg = random.randint(0,20) if random.randint(0,1) else -random.randint(0,20)
        #image = Image.fromarray(img)
        batchP[ind] = misc.imrotate(img,dg)   
#return tf.convert_to_tensor(batchP)
    return batchP









imgsPre = preProcess(imgs)
print(time.time()-s)
