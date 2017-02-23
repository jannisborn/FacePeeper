def GUI_prep(img):
	'''
	This function receives an image uploaded by an user (as np array) and detects the face(s) in the net. 
	There are 3 scenarios:
		1. No face is found: Function returns None. Webserver needs to inform user about that error
		2. One face found: Function returns cropped face of size 112x112 in format: np.shape(img) = (1,112,112,3)
		3. Multiple faces found: Function returns all n faces in format: np.shape(img) = (n,112,112,3)
			Should all be displayed to user and let him choose one.
	 '''
	
	import numpy as np
	import cv2

	# Error Handling
	if not isinstance(img,(np.ndarray,np.generic)):
		print('Please insert a numpy array.')
		return None

	# If image is greyscale, fill the channels up
	if len(img.shape) == 2:
		imgNew = np.zeros((a.shape[0],a.shape[1],3))
		for ind in range(3):
			imgNew[:,:,ind] = img
		img = imgNew


	# Set up classifier and detect images	
	faceDec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	cvImg = cv2.imread(img)
	cvImgBW = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faceCoords = faceDec.detectMultiScale(cvImgBW, scaleFactor=1.3, minNeighbor=5,minSize=(60,60))

	# Handle three scenarios
	if faceCoords:
		faceImgs = np.zeros([len(faceCoords),112,112,3])
		for ind,(x,y,w,h) in enumerate(faceCoords):
			face = img[y:y+h,x:x+w,:]
			faceImgs[ind,:,:,:] = cv2.resize(face,(112,112,3))
		return faceImgs
	else:
		return None





	
