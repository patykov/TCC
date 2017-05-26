#!/usr/bin/env python

import numpy as np
import sys
import os


def mapRGBTrainFrames(datasetDir, classInd, mapFilePath, outputMapFile):
	videoNames = getVideosNames(mapFilePath)
	videoNames = [v.split(' ')[0] for v in videoNames]

	file = open(outputMapFile, 'a')

	for video in videoNames:
		action = video.split('/')[0]
		# -1 -> Because in the .txt file the classes are [0, 101] and cntk only acceps [0,100]
		action_id = int([i for i in classInd.keys() if classInd[i]==action][0]) - 1

		# Video path
		videoPath = os.path.join(datasetDir, video)
		# Frames path
		framesPath = [os.path.join(videoPath, f) for f in os.listdir(videoPath)]
		for f in framesPath:
			file.write('{}\t{}\n'.format(f, action_id))
	file.close()

def mapRGBTestFrames(datasetDir, classInd, mapFilePath, mapFilesDir):
	videoNames = getVideosNames(mapFilePath)
	num = 0

	#Creating a directory for the map files
	if not os.path.exists(mapFilesDir):
		os.mkdir(mapFilesDir)

	for video in videoNames:
		action = video.split('/')[0]
		# -1 -> Because in the .txt file the classes are [0, 101] and cntk only acceps [0,100]
		action_id = int([i for i in classInd.keys() if classInd[i]==action][0]) - 1

		# Video path
		videoPath = os.path.join(datasetDir, video)
		#Frames path
		framesPath = [os.path.join(videoPath, f) for f in os.listdir(videoPath)]
		ids = np.linspace(0, len(framesPath), num=25, dtype=np.int32, endpoint=False)
		
		selected_frames = sorted([framesPath[i] for i in ids])
		
		videoMapFile = os.path.join(mapFilesDir, 'test_map01_RGB{}_{}.txt'.format(num, action_id))
		num+=1
		with open(videoMapFile, 'a') as file:
			for f in selected_frames:
				file.write('{}\t{}\n'.format(f, action_id))


def mapOFTrainFrames(datasetDir, classInd, mapFilePath):
	videoNames = getVideosNames(mapFilePath)

	mapFilePath = os.path.join(os.path.dirname(datasetDir), 'train_map_OF.txt')
	file = open(mapFilePath, 'a')

	for video in videoNames:
		action = video.split('/')[0]
		action_id = [i for i in classInd.keys() if classInd[i]==action]

		for uv in os.listdir(datasetDir):
			# Video path
			videoPath = os.path.join(datasetDir, uv, video)
			# Frames path
			framesPath = [os.path.join(videoPath, f) for f in os.listdir(videoPath)]
			for f in framesPath:
				file.write('{}\t{}\n'.format(f, action_id))

	file.close()


def mapOFTestFrames(datasetDir, classInd, mapFilePath):
	videoNames = getVideosNames(mapFilePath)
	mapFilesDir = os.path.join(os.path.dirname(datasetDir), 'TestMapFiles_FLOW')

	for video in videoNames:
		action = video.split('/')[0]
		action_id = [i for i in classInd.keys() if classInd[i]==action]

		mapFilePath = os.path.join(mapFilesDir, 'test_map_{}.txt'.format(action_id))
		file = open(mapFilePath, 'a')

		for uv in os.listdir(datasetDir):
			# Video path
			videoPath = os.path.join(datasetDir, uv, video)
			# Frames path
			framesPath = [os.path.join(videoPath, f) for f in os.listdir(videoPath)]
			
			ids = np.linspace(0, len(frames), num=25, dtype=np.int32, endpoint=False)
			for i in ids:
				file.write('{}\t{}\n'.format(framesPath[i], action_id))

		file.close()


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
			classInd[int(key)] = val
	return classInd


if __name__ == '__main__':
	RGBdatasetDir = sys.argv[1]
	# OFdatasetDir = sys.argv[2]
	configDir = sys.argv[2]

	classInd = getClassInd(configDir)
	
	# Using Split 1
	trainFilePath = os.path.join(configDir, 'trainlist01.txt')
	testFilePath = os.path.join(configDir, 'testlist01.txt')
	outputTrainPath = os.path.join(os.path.dirname(RGBdatasetDir), 'train_map01_RGB.txt')
	outputTestDir = os.path.join(os.path.dirname(RGBdatasetDir), 'TestMapFiles01_RGB')

	#RGB
	mapRGBTrainFrames(RGBdatasetDir, classInd, trainFilePath, outputTrainPath)
	mapRGBTestFrames(RGBdatasetDir, classInd, testFilePath, outputTestDir)
	#OF
	# mapOFTrainFrames(OFdatasetDir, classInd, trainFilePath)
	# mapOFTestFrames(OFdatasetDir, classInd, testFilePath)
