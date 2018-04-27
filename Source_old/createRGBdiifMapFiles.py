import argparse
import itertools
import numpy as np
import os
import re
import sys
from PIL import Image

def createTrainFile(map_file, dataDir, image_width, image_height, stack_length, 
					channel_count, mapDir=None):
	video_files = []
	targets = []
	with open(map_file, 'r') as file:
		for row in file:
			[video_path, label] = row.replace('\n','').split(' ')
			video_files.append(os.path.join(dataDir, video_path))
			targets.append(int(label)-1)
	
	if not os.path.exists(mapDir):
		os.mkdir(mapDir)
	mapFilesPath = []
	for i in range(5):
		new_mapPath = os.path.join(mapDir, 'trainMap_{}.txt'.format(i+1))
		mapFilesPath.append(new_mapPath)
		# Creating an empty file
		open(new_mapPath, 'w').close()

	mapFilesText = ['']*channel_count
	count = 0
	for (videoFullPath, label) in zip(video_files, targets):
		frames = sorted(os.listdir(videoFullPath), key=lambda x: int(x.split('.')[0]))
		for frameId in range(0, len(frames)-stack_length):
			frameStack = frames[frameId:frameId+stack_length]
			pathFrameStack = [os.path.join(videoFullPath, f) for f in frameStack]

			for i, f in enumerate(mapFilesText):
				mapFilesText[i] += '{}\t{}\t{}\n'.format(count, pathFrameStack[i], label)
			count+=1
			
			if (count%1000 == 0):
				for i, f in enumerate(mapFilesText):
					with open(mapFilesPath[i], 'a') as file:
						file.write(f)
				mapFilesText = ['']*channel_count
				print(count)

	for i, f in enumerate(mapFilesText):
		with open(mapFilesPath[i], 'a') as file:
			file.write(f)

def getTestClass(row, classMap):
	lineClass = row.split('/')[0]
	label = classMap[lineClass]
	return row.replace('\n', ''), label
		
def createTestFile(map_file, classMapFile, dataDir, image_width, image_height, stack_length, 
					channel_count, mapDir=None):
	# Getting class id for test files
	classMap = dict()
	with open(classMapFile, 'r') as file:
		for line in file:
			[label, className] = line.replace('\n', '').split(' ')
			classMap[className] = label
					
	video_files = []
	targets = []
	with open(map_file, 'r') as file:
		for row in file:
			video_path, label = getTestClass(row, classMap)
			video_files.append(os.path.join(dataDir, video_path))
			targets.append(int(label)-1)
	
	if not os.path.exists(mapDir):
		os.mkdir(mapDir)
	mapFilesPath = []
	for i in range(5):
		new_mapPath = os.path.join(mapDir, 'testMap_{}.txt'.format(i+1))
		mapFilesPath.append(new_mapPath)
		# Creating an empty file
		open(new_mapPath, 'w').close()

	mapFilesText = ['']*channel_count
	count = 0
	for (videoFullPath, label) in zip(video_files, targets):
		frames = sorted(os.listdir(videoFullPath), key=lambda x: int(x.split('.')[0]))
		ids = np.linspace(0,len(frames[:-stack_length]),num=25,dtype=np.int32,endpoint=False)
		frameStack = [frames[frameId:frameId+stack_length]	for frameId in ids]
		pathFrameStack = [[os.path.join(videoFullPath, f) for f in fs] for fs in frameStack]

		for fs in pathFrameStack:
			for i, f in enumerate(mapFilesText):
				mapFilesText[i] += '{}\t{}\t{}\n'.format(count, fs[i], label)
			count+=1

		if (count%1000 == 0):
			for i, f in enumerate(mapFilesText):
				with open(mapFilesPath[i], 'a') as file:
					file.write(f)
			mapFilesText = ['']*channel_count
			print(count)

	for i, f in enumerate(mapFilesText):
		with open(mapFilesPath[i], 'a') as file:
			file.write(f)

	
if __name__ == '__main__':
	dataDir		 = "F:/TCC/Datasets/UCF-101_rgb"
	splitDir     = "F:/TCC/Datasets/UCF-101_splits"
	trainMapFile = os.path.join(splitDir, "trainlist01.txt")
	testMapFile	 = os.path.join(splitDir, "testlist01.txt")
	classMapFile = os.path.join(splitDir, "classInd.txt")
	newDirForMap = None #"/hdfs/pnrsy/t-pakova"
	mapDir		 = "F:/TCC/Datasets/RGBdiff_mapFiles"
	
	
	# train_reader = createTrainFile(trainMapFile, dataDir, 224, 224, 5, 5, mapDir=mapDir)
	test_reader = createTestFile(testMapFile, classMapFile, dataDir, 224, 224, 5, 5, mapDir=mapDir)
