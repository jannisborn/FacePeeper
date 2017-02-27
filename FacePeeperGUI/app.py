from flask import Flask, render_template, session, request
from threading import Timer

import flask

import skimage.io
import numpy as np
# import matplotlib.pyplot as plt
import time
import io
import backend.mockup as mockup
from backend.GUI_prep import *

async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = '939fkj3kwlsk4958204kfjnkl39f9Ixne9l39((d'
global cleanUpInterval 
cleanUpInterval = 20 #minutes


global currentlyClassified
currentlyClassified = {} #dict of all image that are currently classified
# they get deleted between cleanUpInterval or 2*cleanUpInterval minutes
global nextToRemove, freshImages
nextToRemove = []
freshImages  = []

def cleanupImageBase():
    """deletes images that where uploaded at least x minutes ago"""
    global nextToRemove, freshImages, currentlyClassified, cleanUpTimer

    for imageHash in nextToRemove:
        del currentlyClassified[imageHash]
        print("in the loop")

    nextToRemove = freshImages
    freshImages = []
    print("cleaned up")

    cleanUpTimer = Timer(60*cleanUpInterval, cleanupImageBase)
    cleanUpTimer.start()


def pairInts(intA, intB):
    """pais two ints uniquly into one"""
    return int(0.5*(intA + intB)*(intA + intB +1)+intB)

@app.route('/')
def index():
    if not 'sessionID' in session:
        print("made new sessionID")
        sessionID = int(time.time()*100%1000000)
        session['sessionID'] = sessionID
    # if you visit the site for the first time, 
    # you get an id so we know which images are yours
    return render_template('index.html')


@app.route('/api/classifyImage/<imageID>', methods=['POST'])
def classifyImage(imageID):
    """classify and save the uploaded image"""
    if not 'sessionID' in session:
        answer = {'message': 'there is no request for this session'}
        resp = flask.jsonify(answer)
        resp.status_code = 400
        return resp

    imageHash = pairInts(session['sessionID'], int(imageID))

    img = skimage.io.imread(request.files['file'])

    imgCropped = GUI_prep(img)
    if(imgCropped is None):
        resp = flask.jsonify({'message': 'We could not detect exactly one face in your image'})
        resp.status_code = 400
        return resp

    currentlyClassified[imageHash] = imgCropped 
    # save the image so that we can retrain with it if the user corrects our prediction
    freshImages.append(imageHash)
    # rember the hash, so that we can delete it if it is to old

    label = mockup.classifyImage(imgCropped)
    # at the moment our classifier doesn't actually work, so it is mocked up

    answer = {'label': label}

    return flask.jsonify(answer)

@app.route('/api/correctClassification/<imageID>', methods=['POST'])
def correctClassification(imageID):
    """lets the user post a correct name for the image"""
    mistake = checkIfCorrectRequestedImage(imageID)
    if mistake is not None:
        print("no Request in correctClassification")
        return mistake

    sessionID = session["sessionID"]
    imageHash = pairInts(sessionID, int(imageID))

    newName = request.form["newName"]

    # get the picture that he wants updated
    pic = currentlyClassified[imageHash]


    # the backend retraining doesn't actually work
    mockup.updateClassification(pic, newName)
    return flask.jsonify({"message":"py says success"})



@app.route('/api/actorList')
def getActorList():
    """return list of all actors"""
    return flask.jsonify(mockup.getActorList())

@app.route('/api/actorInfo/<name>')
def actorInfoByName(name):
    """given an actor name as specified by the getActorList give back info as plain text"""
    return "We don't have any info text available for anyone."

@app.route('/api/getPreProcessedImg/<imageID>')
def getImage(imageID):
    """get image from the currently classified images, 
    the server created session id plus the user created image id are used to identify it"""

    mistake = checkIfCorrectRequestedImage(imageID)
    if mistake is not None:
        return mistake


    sessionID = session["sessionID"]
    imageHash = pairInts(sessionID, int(imageID))

    pic = currentlyClassified[imageHash]

    picSaved = io.BytesIO()
    skimage.io.imsave(picSaved, pic)

    #putting the reader at the begining of the file:
    picSaved.seek(0)

    print('sending')
    return flask.send_file(picSaved, mimetype="image/jpeg", cache_timeout=0.1)


def checkIfCorrectRequestedImage(imageID):
    '''helper function to see that the image exists'''
    if not 'sessionID' in session:
        answer = {'message': 'there is no request for this session'}
        resp = flask.jsonify(answer)
        resp.status_code = 400
        return resp
    sessionID = session['sessionID']
    imageHash = pairInts(sessionID, int(imageID))

    if (not imageHash in currentlyClassified):
        answer = {'message': 'the requested image does not exist'}
        resp = flask.jsonify(answer)
        resp.status_code = 400
        return resp
    return None


if __name__ == '__main__':
    print('in my file')
    cleanupImageBase()
    Flask.run(app, debug=False)
    cleanUpTimer.cancel()
    print("ended gracefully")


