import numpy as np
import glob
from scipy import misc
import os
import time

# Read in Images
numImgs = 50312
imgs = np.empty([numImgs,112,112,3])
filenames = os.listdir("C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/")

for ind, filename in enumerate(glob.glob('C:/Users/Jannis/Dropbox/GitHub/FacePeeper/Data/*.jpg')):
#for ind, filename in enumerate(glob.glob('/Users/jannis/Dropbox/github/FacePeeper/Data/*.jpg')):
	if ind < numImgs:
		imgs[ind,:,:,:] = misc.imread(filename) 
print('Images stored')

np.savetxt('data.txt',imgs)