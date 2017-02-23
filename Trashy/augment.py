import numpy as np
import os
from scipy import misc

root = 'C:/Users/Jannis/Dropbox/GitHub/FacePeeper/'






def augment(batch,mu=0,sigma=0.01):
        '''
        This augmentation function is inspired by: Krizhevsky et al. (2012): ImageNet 
            Classification with Deep Convolutional Neural Networks
        This function receives a batch of images (np.array of size [BatchSize,112,112,3]) 
        and performs the following augmentation steps (on the fly, for each img):
            1. Color Augmentation based on channel-wise PCA of entire dataset.
                VarianceIncreaser + NoiseAdder
            2. Binarizes about flipping vertically
            3. Rotation within range of [-25,25] degree
        The optional parameters mu and sigma define the average and the spread in the noise added
        to each RGB channel  '''
        

        batchSize = batch.shape[0]

        # Allocate array for augmented images 
        batchP = np.empty([batchSize,112,112,3],dtype=np.uint8)

        # Restore PCA results that has been performed beforehand on entire dataset
        eigenvalues = np.loadtxt(root+'eigenvalues.txt')
        eigenvectors = np.loadtxt(root+'eigenvectors.txt')
        
        # generate stochastic noise (alpha samples)
        samples = np.random.normal(mu,sigma,[batchSize,3])
        augmentation = samples * eigenvalues # scale by eigenvalue
        
        # augment every image
        for ind,img in enumerate(batch):
            # RGB augmentation via PCA (increase variance, tune luminance+color invariance)
            noise = np.dot(eigenvectors,augmentation[ind])
            img = img + noise # Add color perturbation to image

            # Flip horizontally (eventually)
            img = np.fliplr(img) if np.random.randint(2) else img

            # Rotate randomly 
            dg = random.randint(0,20) if random.randint(0,1) else -random.randint(0,20)
            batchP[ind] = misc.imrotate(img, dg) 
                
        return batchP



path = root+'Plots/normPlot/'
imgs = np.empty([4,112,112,3])
imgs[0] = misc.imread(path+'a.jpg')
imgs[1] = misc.imread(path+'b.jpg')
imgs[2] = misc.imread(path+'c.jpg')
imgs[3] = misc.imread(path+'d.jpg')


imgs = augment(imgs)

for ind,img in enumerate(imgs):
    misc.imsave(path+ind+'.jpg',img)