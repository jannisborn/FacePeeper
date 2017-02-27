import numpy as np
import random
import re
#import matplotlib.pyplot as plt


def faceCrop(img):
 	return img[:112,:112,:]

def classifyImage(img):
	i = random.randrange(0, len(allNames))
	return allNames[i]

def getActorInfo(name):
	return "Deepika Padukone (pronounced [d̪iːpɪkaː pəɖʊkoːɳ]; born 5 January 1986) is an Indian film actress. One of the highest-paid actresses in the world, Padukone is the recipient of several awards, including three Filmfare Awards. She features in listings of the nation's most popular and attractive personalities."

def getActorDicts(filename="backend/identities.txt"):
	list_file = open(filename, "r", encoding='utf-8-sig')
	lines = list_file.readlines()
	list_file.close()
	number2name = {}
	name2number = {}
	for l in lines:
		m = re.match(r"(\d\d\d): (.*)", l)
		if m is not None:
			number, name = m.groups()
			if number != "" and name != "":
				number = int(number)
				number2name[number] = name
				name2number[name]	= number
	return number2name, name2number

def getActorList():
	num2name, name2number = getActorDicts()
	names = list(name2number.keys())
	return names

def updateClassification(img, label):
	print(label)
	#plt.ion()
	#plt.imshow(img)
	#plt.show()
	#plt.pause(1)


allNames = getActorList()
