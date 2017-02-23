for epoch in range(epochs):
        
        print('epoch: ', epoch)
        epochInds = np.random.permutation(self.numTrainImgs)        

        for batchNumber in range(self.numTrainImgs//batchSize):
            print('batch_nr: ', batchNumber)

            batchInds = epochInds[(batchNumber*batchSize):(batchNumber+1)*batchSize]

            trainingImages, trainingLabels = celeb.createBatch(batchInds, 'trainData')
            # Data augmentation
            trainingImages = preProcess(trainingImages)

            stLR = 0.0000001
            #learningRate = tf.train.exponential_decay(starterLearningRate, globalStep, 10e3, 0.96)

            #crossEntr[step], accur[step], _ = session.run([crossEntropy, accuracy, trainStep],
            #                        feed_dict = {images: trainingImages, desired: trainingLabels, lr: stLR})

            #print('Accuracy: ', accur[step])
            prediction = tf.argmax(tf.nn.softmax(logits), 1)
            des = desired
            best, des = session.run([prediction, des], feed_dict = {images: trainingImages, desired: trainingLabels, lr: stLR})
            print('Best prediction: ', best, ' - Desired: ', des)

            if (step % 25 == 0 and step != 0) or step == trainingSteps-1:
                saver.save(session, "./resnet.chkp", step)
                
            step += 1

            #f = plt.figure()
            #x1 = np.linspace(25,epochs,epochs-25)
            #plt.plot(x1,accur[25:])
            #f.savefig('CELEBRITIES.png')
 
with tf.Session() as session:
    saver = tf.train.Saver()
    # We restore the weights saved in training and test them on test data.
    saver.restore(session, "./resnet.chkp-" + str(trainingSteps-1))    
    
    accuracies = []
    for r in range(self.numTestImgs//batchSize:

        batchInds = testInds[(r*batchSize):(r+1)*batchSize]
        testImages, testLabels = celeb.createBatch(batchInds, 'testData')
        crossEntr, accur = session.run([crossEntropy, accuracy], feed_dict = {images: testImages, desired: testLabels})
        accuracies.append(accur)

    print(np.mean(accuracies))