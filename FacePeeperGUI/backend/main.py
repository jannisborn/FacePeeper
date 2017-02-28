import numpy as np
import random
import re
from backend.GUI_prep import *
from backend.genderization import *
#import matplotlib.pyplot as plt

def getActorDicts(filename="backend/identities.txt"):
	list_file = open(filename, "r", encoding='utf-8-sig')
	lines = list_file.readlines()
	list_file.close()
	number2name = {}
	name2number = {}
	for l in lines:
		m = re.match(r"(\d{1,5}): (.*)", l)
		if m is not None:
			number, name = m.groups()
			if number != "" and name != "":
				number = int(number)
				number2name[number] = name
				name2number[name]	= number
	print(number2name)
	return number2name, name2number

def getActorList():
	global name2number
	names = list(name2number.keys())
	return names

def cropp(img):
	"""cropes the image to just the face"""
	return GUI_prep(img)

def classify(img):
	"""classifies the image"""
	accuracy = genderization(img)[0]
	label = np.argmax(accuracy)
	return "{} ({:2.0%})".format(number2name[label], accuracy[label])
	

def updateClassification(img, txtLabel):
	intTrainLabel = name2number[txtLabel] #TODO handle exception
	accuracy = genderization(img, intTrainLabel)[0]
	label = np.argmax(accuracy)

	return "{} ({:2.0%})".format(number2name[label], accuracy[label])


number2name, name2number = getActorDicts()
print(number2name)
