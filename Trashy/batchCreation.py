    def createBatch(self, indices, batchType = ''):
        '''
        This function receives the indices pointing to the list that contains the names of all
            training/testing images. It loads these images and thus creates the next batch
        Images and labels are returned as np.arrays
        '''
        
       batchSize = len(indices)

       # Define dataset (train/test), error handling
        if batchType == 'trainData':
            path = dir_train
            files = self.trainFiles
        elif batchType == 'testData':
            path = dir_test
            files = self.testFiles
        else:
            raise ValueError("Please call createBatch function with 'trainData' or 'testData' as batchType")

        # Allocate space (append is slow...)
        images = np.empty([batchSize,112,112,3])
        labels = np.empty(batchSize,dtype=int)

        # Load images and labels one by one
        for counter,imgInd in enumerate(indices):
            images[counter,:,:,:] = misc.imread(path + files[imgInd])
            labels[counter] = int(files[imgInd][:3])

        # The images have been normalized before saving. However misc.imsave scales everything to [0,255]
        # The range of zscores after normalization was suprisingly close to [-2,2]. We therefore rescale [0,255] --> [-2,2]
        images = (4*images)/255
        images -= 2
        return images, labels

