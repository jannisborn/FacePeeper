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
    with tf.Session():
        # generate Alpha Samples (to add noise)
        samples = np.random.normal(mu,sigma,[batchSize,3])
        augmentation = samples * eigenvalues # scale by eigenvalue
        for ind,img in enumerate(batch):

            # RGB augmentation via PCA (increase variance, tune luminance+color invariance)
            noise = np.dot(eigenvectors,augmentation[ind])
            img = img + noise # Add color perturbation to image



            #Flip Veritcally?
            img = tf.random_flip_left_right(tf.convert_to_tensor(img))
            #img = tf.convert_to_tensor(img)

            # Rotate randomly 
            dg = random.randint(0,20) if random.randint(0,1) else -random.randint(0,20)
            batchP[ind] = misc.imrotate(img,dg)   
            
    return tf.convert_to_tensor(batchP)