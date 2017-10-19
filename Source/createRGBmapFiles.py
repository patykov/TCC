import argparse
import itertools
import numpy as np
import os
import re
import sys
from PIL import Image
					
class createMapFile(object):

	def __init__(self, map_file, dataDir, image_width, image_height, channel_count, 
					label_count, max_epochs, minibatch_size, is_training=True, 
					classMapFile=None, mapDir=None, newImgDir=None):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.map_file		 = map_file
		self.label_count	 = label_count
		self.width			 = image_width
		self.height			 = image_height
		self.sequence_length = 1
		self.is_training	 = is_training
		self.channel_count	 = channel_count
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList		 = [None]*self.label_count
		self.dataDir		 = dataDir
		self.newImgDir		 = newImgDir
		self.mapDir			 = mapDir

		if not self.is_training:
			self.sequence_length = 250
			self.getClassDict(classMapFile)
		
		with open(map_file, 'r') as file:
			for row in file:
				if self.is_training:
					[video_path, label] = row.replace('\n','').split(' ')
				else:
					video_path, label = self.getTestClass(row)
				video_path = os.path.join(dataDir, video_path)
				self.video_files.append(video_path)
				self.targets.append(int(label)-1)
				if self.myAuxList[int(label)-1] == None:
					self.myAuxList[int(label)-1] = [len(self.targets)-1]
				else:
					self.myAuxList[int(label)-1].append(len(self.targets)-1)

		self.indices = np.arange(len(self.video_files))
		self.createMap(max_epochs, minibatch_size)

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
		
	def size(self):
		return len(self.video_files)
			
	def has_more(self):
		if self.batch_start < self.size():
			return True
		return False

	def reset(self):
		self.batch_start = 0
		if self.is_training:
			self.groupByTarget()

	def groupByTarget(self):
		workList = self.myAuxList[::]
		if self.is_training:
			for x in workList:
				np.random.shuffle(x)
			np.random.shuffle(workList)
		grouped = list(itertools.zip_longest(*workList))
		np.random.shuffle(grouped)
		self.indices = [x for x in itertools.chain(*grouped) if x != None]
	
	def next_minibatch(self, batch_size):
		'''
		Return a mini batch of sequence frames and their corresponding ground truth.
		'''
		batch_end = min(self.batch_start + batch_size, self.size())
		current_batch_size = batch_end - self.batch_start
		
		if current_batch_size < 0:
			raise Exception('Reach the end of the training data.')
		
		allFrames, allLabels = [], []
		for idx in range(self.batch_start, batch_end):
			index = self.indices[idx]
			allFrames.append(self._select_features(self.video_files[index]))
			allLabels.append(self.targets[index])
		self.batch_start += current_batch_size
		
		return allFrames, allLabels, current_batch_size

	def _select_features(self, video_path):
		'''
		Select a sequence of frames from video_path and return them as a Tensor.
		'''
		videoFullPath = os.path.join(self.dataDir, video_path)

		frames = os.listdir(videoFullPath)
		selectedFrames = []

		if self.is_training:
			selectedFrame = np.random.choice(frames)
			pathFrameStack = [os.path.join(videoFullPath, selectedFrame)]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0,len(frames),num=length,dtype=np.int32,endpoint=False)
			pathFrameStack = sorted([os.path.join(videoFullPath, frames[i]) for i in ids],
										key=lambda x: int(x.split('\\')[-1].split('.')[0]))

		return pathFrameStack

		
	def createMap(self, max_epochs, minibatch_size):
		mapFilePath = self.createMapPath()
		count = 0
		
		for epoch in range(max_epochs): # loop over epochs
			mapFileText = ''
			self.reset()
			while self.has_more():		# loop over minibatches in the epoch
				frames, labels, current_minibatch = self.next_minibatch(minibatch_size)
				for mb_fs, mb_l in zip(frames, labels):
					for fs in mb_fs:
						mapFileText += '{}\t{}\t{}\n'.format(count, fs, mb_l)
						count+=1
					
			with open(mapFilePath, 'a') as file:
				file.write(mapFileText)
			
			print('Finished epoch {}'.format(epoch))
			
	def createMapPath(self):
		if not os.path.exists(self.mapDir):
			os.mkdir(self.mapDir)
		split_id = int(re.findall(r'\d+', self.map_file)[0])
		if self.is_training: 
			new_mapPath = os.path.join(self.mapDir, 'trainMap_{}.txt'.format(split_id))
		else:
			new_mapPath = os.path.join(self.mapDir, 'testMap_{}.txt'.format(split_id))
		# Creating an empty file
		open(new_mapPath, 'w').close()
		return new_mapPath

						
					
if __name__ == '__main__':
	dataDir		 = "F:/TCC/Datasets/UCF-101_rgb"
	splitDataDir = "F:/TCC/Datasets/ucfTrainTestlist"
	trainMapFile = os.path.join(splitDataDir, "trainlist01.txt")
	testMapFile	 = os.path.join(splitDataDir, "testlist01.txt")
	classMapFile = os.path.join(splitDataDir, "classInd.txt")
	newDirForMap = None #"/hdfs/pnrsy/t-pakova"
	mapDir		 = "F:/TCC/Datasets/RGB_mapFiles_forLaterXforms"
	
	# Model dimensions
	num_classes	 = 101
	minibatch_size = 256
	
	print('Creating train map files')
	max_epochs = 537
	train_reader = createMapFile(trainMapFile, dataDir, 224, 224, 3, 
					num_classes, max_epochs, minibatch_size, is_training=True, 
					classMapFile=None, mapDir=mapDir, newImgDir=newDirForMap)
					
	# print('Creating test map files')
	# test_reader = createMapFile(testMapFile, dataDir, 224, 224, 3, 
					# num_classes, 1, minibatch_size, is_training=False, 
					# classMapFile=classMapFile, mapDir=mapDir, newImgDir=newDirForMap)

	
