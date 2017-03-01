# FacePeeper
A Deep Residual Convolutional Neural Network as Gender Classifier on an Interactive Webserver. The network is implemented in [Tensorflow](https://www.tensorflow.org/) an optimized with [ADAM](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) 

## Project Specifications 
- University of Osnabrück
- Winter Term 2016/17
- Course: Introduction to Artificial Neural Networks with Tensorflow
- Contributors: Clemens Hutter, Michele Pariani, Jannik Steinmann, Jannis Born


## Requirements
- Full: 
  * Python 3.x
  * Tensorflow 1.x (GPU access highly recommended)
  * openCV
  * Scrapy
  * C++ Compiler (OS X, Linux or Visual Studios on Windows)
  * Webserver access
  * Scipy
- Partly (functionality verification)
  * Python 3.x
  * Tensorflow 1.x
  * Scipy
  
 
 
## Overview
For this project we crawled pictures from ca. 400 celebrities from the Image Movie Database (IMDB). These images were 
face-cropped and initially used to train a 33-layer Residual Convolutional Neural Network (according to 
[He et al., Deep Residual Learning for Image Recognition (2015) ](https://arxiv.org/pdf/1512.03385.pdf)) to differentiate
the identities of these 400 celebrities. Due to a) the complexity of this task and b) time constraints, we simplified the 
architecture to a 9-layer residual CNN trained on a cross-individual gender classification. On the frontend, we provide
a webserver on which a user verify the network's gender prediction of an arbitrary uploaded image. In case of wrong 
classification the user has the option to provide the correct label and retrain the network.

## Guideline
- The **Crawler** folder contains the code necessary to crawl the images from IMDB
- The core element of this project, a class called *RESNET()*, that implements the Residual Convolutional Neuronal Network, can be found in the main directory within **residualCNN.py**.
- An extensive report of the project can be found in *REPORT.pdf* 
- The network is implemented in Tensorflow (as class) and allows access from the folders **GenderClassifier**, 
**IdentityClassifier** and **MNIST_Classifier**. These three folders essentially vary only according to the task they solve
by means of the CNN in residualCNN.py. They have very similar structures, each containing a Trainer.py file 
(that trains on the particular task from scratch) and a Tester.py file (that restores our pretrained weights and tests model
performance). They also contain performance visualization (.pdf) based on the raw performance files (.txt). The initial task,
identitiy classification could not be solved properly although learning slowly began after some time (and that only after reducing
depth of the network). We verified the soundness of the network by classifying the MNIST dataset where we achieved a test 
accuracy of 99.56%. Subsequently, we used a subset of the 400-class celebrity dataset to train the network on gender classifcation
task where we achieve 96.4% on the test dataset.
  * To restore the trained networks the weights can be found on [Dropbox] (https://www.dropbox.com/sh/sfc1he6spgq4fv9/AAAZ2nrfESVxIHC6rQhVAfkma?dl=0)
  (files too large for Git)
  * For smooth execution, please place the weight file in the respective folder (e.g. weightsIdentity.ckpt in IdentityClassifier)
  * Please consider, that for the GenderClassification we used once a traditional and once an augmented dataset (according to
  [Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks (2012)] 
  (https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf)). You can
  specify which version/weights you want to use via a Boolean at the beginning of the training file.
- The **FacePeeperGUI** folder provides the code for the interactive frontend where a user can a) upload and classify any image
(given a single face is found) and b) correct misclassifications of the network. It is implemented in a combination
of Python, HTML and JavaScript and the webserver can be made temporarily available via Amazon Web Services (AWS) upon request.
- The **Helper** folder contains a preProcessing file (normalization, PCA of RGB augmentation), the results of the PCA, 
a MATLAB function used to crop the crawled images, a plot visualizing the effect of image normalization and augmentation, 
and some backup files for the frontend.
- The folders **TrainData** and **TestData** contain selections of the crawled images, please leave them untouched.

  
  
  
  
  
  
  
