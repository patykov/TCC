import argparse
import numpy as np
import os
import re
import sys
from PIL import Image

def createTrainFile(map_file, dataDir, image_width, image_height, stack_length, 
					num_classes, channel_count, mapDir=None):
	video_files = []
	targets = []
	with open(map_file, 'r') as file:
		for row in file:
			[video_path, label] = row.replace('\n','').split(' ')
			video_path = re.search('/(.*).avi', video_path).group(1)
			video_files.append(os.path.join(dataDir, 'u', video_path))
			targets.append(int(label)-1)
	
	if not os.path.exists(mapDir):
		os.mkdir(mapDir)
	mapFilesPath = []
	for i in range(20):
		new_mapPath = os.path.join(mapDir, 'trainMap_{}.txt'.format(i+1))
		mapFilesPath.append(new_mapPath)
		# Creating an empty file
		open(new_mapPath, 'w').close()

	mapFilesText = ['']*channel_count
	count = 0
	for (videoFullPath, label) in zip(video_files, targets):
		frames = os.listdir(videoFullPath)
		for frameId in range(0, len(frames)-stack_length, 2):
			selectedFrames = []
			frameStack = frames[frameId:frameId+stack_length]
			pathFrameStack = [[os.path.join(videoFullPath, f) for f in frameStack]]

			for fs in pathFrameStack:
				for i, f in enumerate(mapFilesText):
					if i%2==0:
						mapFilesText[i] += '{}\t{}\t{}\n'.format(count, fs[int(i/2)], label)
					else:
						mapFilesText[i] += '{}\t{}\t{}\n'.format(count, fs[int(i/2)].replace('\\u\\', '\\v\\'), label)
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
	dataDir		 = "F:/TCC/Datasets/UCF-101_opticalFlow"
	trainMapFile = os.path.join(dataDir, "trainlist01.txt")
	testMapFile	 = os.path.join(dataDir, "testlist01.txt")
	classMapFile = os.path.join(dataDir, "classInd.txt")
	newDirForMap = None #"/hdfs/pnrsy/t-pakova"
	mapDir		 = "F:/TCC/Datasets/OF_mapFiles-half"
	
	# Model dimensions
	stack_length = 10
	num_classes	 = 101

	
	print('Creating train map files')
	train_reader = createTrainFile(trainMapFile, dataDir, 224, 224, stack_length, 
					num_classes, 20, mapDir=mapDir)
	# print('Creating test map files')
	# test_reader = createMapFile(testMapFile, dataDir, 224, 224, stack_length, 
					# num_classes, 1, 1, is_training=False, 
					# classMapFile=classMapFile, mapDir=mapDir, newImgDir=dirForNewImg)
