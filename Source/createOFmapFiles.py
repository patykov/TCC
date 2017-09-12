import argparse
import itertools
import numpy as np
import os
import re
import sys

# Define the reader for both training and evaluation action.
class createMapFiles(object):

	def __init__(self, map_file, dataDir, stack_length, label_count, is_training=True, 
					newDir=None, classMapFile=None):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.is_training	 = is_training
		self.stack_length	 = stack_length
		self.channel_count	 = 2*stack_length
		self.flowRange		 = 40.0
		self.imageRange		 = 255.0
		self.sequence_length = 1
		self.video_files	 = []
		self.targets		 = []
		self.dataDir		 = dataDir
		self.newDir          = newDir if newDir!=None else dataDir

		if not self.is_training:
			self.sequence_length = 250
			self.getClassDict(classMapFile)
		
		with open(map_file, 'r') as file:
			for row in file:
				if self.is_training:
					[video_path, label] = row.replace('\n','').split(' ')
				else:
					video_path, label = self.getTestClass(row)
				video_path = re.search('/(.*).avi', video_path).group(1)
				self.video_files.append(video_path)
				self.targets.append(int(label)-1)

		self.createFiles()

	def getClassDict(self, classMapFile):
		# Getting class id for test files
		self.classMap = dict()
		with open(classMapFile, 'r') as file:
			for line in file:
				[label, className] = line.replace('\n', '').split(' ')
				self.classMap[className] = label

	def getTestClass(self, row):
		lineClass = row.split('/')[0]
		label = self.classMap[lineClass]
		return row.replace('\n', ''), label

	def createFiles(self):
		mapFiles  = ['']*self.channel_count
		allFrames, allLabels = [], []
		count = 0
		for i, video_path in enumerate(self.video_files):
			print(i, video_path)
			frames = os.listdir(os.path.join(self.dataDir, 'v', video_path))
			
			if self.is_training:
				for frame in frames[:-self.stack_length]:
					frameId = frames.index(frame)
					allFrames.append(frames[frameId:frameId+self.stack_length])
					allLabels.append(self.targets[i])
			else:
				length = self.sequence_length/10
				ids = np.linspace(0, len(frames[:-self.stack_length]), num=length, dtype=np.int32, endpoint=False)
				for frameId in ids:
					allFrames.append(frames[frameId:frameId+self.stack_length])
					allLabels.append(self.targets[i])
		
			for stack in allFrames:
				assert(len(mapFiles) == 2*len(stack))
				for i, f in enumerate(mapFiles):
					if i%2==0:
						mapFiles[i] += '{}\t{}\t{}\n'.format(count, os.path.join(self.newDir, 'v.zip@', video_path, stack[int(i/2)]), allLabels[i%2])
					else:
						mapFiles[i] += '{}\t{}\t{}\n'.format(count, os.path.join(self.newDir, 'u.zip@', video_path, stack[int(i/2)]), allLabels[i%2])
				count+=1
			
			if (i%10==0):
				print('yey')
				for i, mapF in enumerate(mapFiles):
					print(mapF)
					if self.is_training:
						fileName = os.path.join(self.newDir, 'trainMap_{}.txt'.format(i))
						with open(fileName, 'a') as file:
							file.write(mapF)
					else:
						fileName = os.path.join(self.newDir, 'testMap_{}.txt'.format(i))
						with open(fileName, 'a') as file:
							file.write(mapF)
				mapFiles  = ['']*self.channel_count
				allFrames, allLabels = [], []
			
		for i, mapF in enumerate(mapFiles):
			if self.is_training:
				fileName = os.path.join(self.newDir, 'trainMap_{}.txt'.format(i))
				with open(fileName, 'a') as file:
					file.write(mapF)
			else:
				fileName = os.path.join(self.newDir, 'testMap_{}.txt'.format(i))
				with open(fileName, 'a') as file:
					file.write(mapF)


if __name__ == '__main__':
	dataDir = "F:/TCC/Datasets/UCF-101_opticalFlow"
	trainMapFile = os.path.join(dataDir, "trainlist01.txt")
	testMapFile  = os.path.join(dataDir, "testlist01.txt")
	classMapFile = os.path.join(dataDir, "classInd.txt")
	newDir = "F:\TCC\Datasets\OF_newMapFiles" #"/hdfs/pnrsy/t-pakova"
	
	# Model dimensions
	stack_length = 10
	num_classes	 = 101

	createMapFiles(trainMapFile, dataDir, stack_length, num_classes, is_training=True, newDir=newDir)
	createMapFiles(testMapFile, dataDir, stack_length, num_classes, is_training=False, classMapFile=classMapFile, newDir=newDir)
	
