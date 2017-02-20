def preProcess(batch):

    '''
    This function receives a batch of images (np.array of size [BatchSize,112,112,3]) 
    and performs the following preprocessing steps (for each img):
        1. Color Augmentation based on channel-wise PCA of entire dataset.
            VarianceIncreaser + NoiseAdder
        2. Binarizes about flipping vertically
        3. Rotation within range of [-25,25] degree
        
    Afterwards it returns a TF Tensor
    '''
    import random
    from PIL import Image
    from scipy import misc

    batchP = np.empty([batch.shape[0],112,112,3],dtype=np.uint8)
    with tf.Session():
        for ind,img in enumerate(batch):

            #ToDo: PCA part

            #Flip Veritcally?
            #img = tf.random_flip_left_right(tf.convert_to_tensor(img))
            img = tf.convert_to_tensor(img)

            # Rotate randomly 
            dg = random.randint(0,20) if random.randint(0,1) else -random.randint(0,20)
            batchP[ind] = np.array(Image.Image.rotate(Image.fromarray(img.eval()),dg))   
    return tf.convert_to_tensor(batchP)