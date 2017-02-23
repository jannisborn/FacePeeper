#  Updated Batch Function! (Wednesday night J)
# Note that I have added indices as a third argument now


    def createBatch(self, indices, batchType = ''):
        '''
        This function receives a batchSize and randomly loads some images from the dataset.
        Images and labels are returned as np.array
        '''
        
        batchSize = len(indices)

        if batchType == 'trainData':
            path = dir_train
            files = self.trainFiles
            num = self.numTrainImgs
        else:
            path = dir_test
            files = self.testFiles
            num = self.numTestImgs

        images = np.empty([batchSize,112,112,3])
        labels = np.empty(batchSize,dtype=int)

        for counter,imgInd in enumerate(indices):
            images[counter,:,:,:] = misc.imread(path + files[imgInd])
            labels[counter] = int(files[imgInd][:3])

        # The images have been normalized before saving. However misc.imsave scales everything to [0,255]
        # The range of zscores after normalization was suprisingly close to [-2,2]. We therefore rescale [0,255] --> [-2,2]
        images = (4*images)/255
        images -= 2
        return images, labels


