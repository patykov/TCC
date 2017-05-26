#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os


def saveMapImages(dirImgName, datasetDir, mapFileName):
	imgsNames = sorted(os.listdir(dirImgName))
	imgsPaths = [os.path.join(dirImgName, name) for name in imgsNames]
	mapFilePath = os.path.join(datasetDir, mapFileName)
	with open(mapFilePath, 'w') as mapFile:
		for imgPath, imgName in zip(imgsPaths, imgsNames):
			label = imgName.split('_')[-1].split('.')[0]
			mapFile.write('{}\t{}\n'.format(imgPath, label))


def extractTrainFrames(datasetDir, configDir, classInd):
	videoNames = getVideosNames(os.path.join(configDir, 'trainlist01.txt'))
	imgsDir = datasetDir+'_trainFrames'
	totalNum = 0

	#Dataset frame directory
	if not os.path.exists(imgsDir):
		os.mkdir(imgsDir)

	for video in videoNames:
		action = video.split('/')[0]
		video = video.split(' ')[0]
		action_id = [i for i in classInd.keys() if classInd[i]==action][0]
		
		#Video path
		videoPath = os.path.join(datasetDir, video)

		cam = cv2.VideoCapture(videoPath)
		success, image = cam.read()
		while success:
			success, image = cam.read()
			if success:
				cv2.imwrite('{}/{}_{}.jpg'.format(imgsDir, totalNum, action_id), image)
				totalNum +=1

	saveMapImages(imgsDir, os.path.dirname(datasetDir), 'train_map.txt')


def extractTestFrames(datasetDir, configDir, classInd):
	videoNames = getVideosNames(os.path.join(configDir, 'testlist01.txt'))
	imgsDir = datasetDir+'_testFrames'
	mapFilesDir = os.path.join(os.path.dirname(datasetDir), 'TestMapFiles')
	totalNum = 0
	videoNum = 0

	#Creating a directory for the frames
	if not os.path.exists(imgsDir):
		os.mkdir(imgsDir)

	#Creating a directory for the map files
	if not os.path.exists(mapFilesDir):
		os.mkdir(mapFilesDir)

	for video in videoNames:
		action = video.split('/')[0]
		actionFramesDir = os.path.join(imgsDir, action)
		if not os.path.exists(actionFramesDir):
			os.mkdir(actionFramesDir)
			
		# Video frames dir
		videoFramesDir = os.path.join(actionFramesDir, video.split('/')[1])
		os.mkdir(videoFramesDir)

		# Video path
		videoPath = os.path.join(datasetDir, video)

		frames = []
		cam = cv2.VideoCapture(videoPath)
		success, image = cam.read()
		while success:
			success, image = cam.read()
			if success:
				frames.append(image)

		ids = np.linspace(0, len(frames), num=25, dtype=np.int32, endpoint=False)
		action_id = [i for i in classInd.keys() if classInd[i]==action][0]
		for i in ids:
			cv2.imwrite('{}/{}_{}.jpg'.format(videoFramesDir, totalNum, action_id), frames[i])
			totalNum +=1

		saveMapImages(videoFramesDir, mapFilesDir, 'test_map{}_{}.txt'.format(videoNum, action_id))
		videoNum +=1


def getVideosNames(filePath):
	videoNames = []
	with open(filePath, 'r') as file:
		for line in file:
			video = line.replace('\n', '')
			videoNames.append(video)
	return videoNames


def getClassInd(configDir):
	filePath = os.path.join(configDir, 'classInd.txt')
	classInd = dict()
	with open(filePath, 'r') as file:
		for line in file:
			(key, val) = line.replace('\n', '').split(' ')
			classInd[int(key)-1] = val
	return classInd


if __name__ == '__main__':
	datasetDir = sys.argv[1]
	configDir = sys.argv[2]

	classInd = getClassInd(configDir)
	
	# Using Split 1
	#extractTestFrames(datasetDir, configDir, classInd)
	extractTrainFrames(datasetDir, configDir, classInd)


	
	