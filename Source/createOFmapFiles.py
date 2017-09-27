import argparse
import itertools
import numpy as np
import os
import re
import sys
from PIL import Image
					
class createMapFile(object):

	def __init__(self, map_file, dataDir, image_width, image_height, stack_length, 
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
		self.multiView		 = self.getMultiView()
		self.stack_length	 = stack_length
		self.channel_count	 = 2*stack_length
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList		 = [None]*self.label_count
		self.dataDir		 = dataDir
		self.newImgDir		 = newImgDir
		self.mapDir			 = mapDir

		# Make sure to dont rewrite the original images
		# assert(self.newImgDir != self.dataDir)
		# if not os.path.exists(self.newImgDir):
			# os.mkdir(self.newImgDir)
		# if not os.path.exists(os.path.join(self.newImgDir, 'v')):
			# os.mkdir(os.path.join(self.newImgDir, 'v'))
		# if not os.path.exists(os.path.join(self.newImgDir, 'u')):
			# os.mkdir(os.path.join(self.newImgDir, 'u'))
		# labels = os.listdir(os.path.join(self.dataDir, 'v'))
		# for l in labels:
			# path_v = os.path.join(self.newImgDir, 'v', l)
			# path_u = os.path.join(self.newImgDir, 'u', l)
			# if not os.path.exists(path_v):
				# os.mkdir(path_v)
			# if not os.path.exists(path_u):
				# os.mkdir(path_u)

	
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
				if self.myAuxList[int(label)-1] == None:
					self.myAuxList[int(label)-1] = [len(self.targets)-1]
				else:
					self.myAuxList[int(label)-1].append(len(self.targets)-1)

		self.indices = np.arange(len(self.video_files))
		self.groupByTarget()
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
	
	def getMultiView(self):
		img1 = lambda x: x.crop((x.size[0]/2 - self.width/2, x.size[1]/2 - self.height/2, 
								x.size[0]/2 + self.width/2, x.size[1]/2 + self.height/2)) # center
		img2 = lambda x: x.crop((0, 0, 
								self.width, self.height)) # top left
		img3 = lambda x: x.crop((0, x.size[1] - self.height, 
								self.width, x.size[1])) # bottom left
		img4 = lambda x: x.crop((x.size[0] - self.width, 0, 
								x.size[0], self.height)) #top right
		img5 = lambda x: x.crop((x.size[0] - self.width, x.size[1] - self.height, 
								x.size[0], x.size[1])) # bottom right
		# Flipped
		img6 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(x.size[0]/2 - self.width/2, x.size[1]/2 - self.height/2, 
								x.size[0]/2 + self.width/2, x.size[1]/2 + self.height/2)) # flip center
		img7 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(0, 0, 
								self.width, self.height)) # flip top left
		img8 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(0, x.size[1] - self.height, 
								self.width, x.size[1])) # bottom left
		img9 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(x.size[0] - self.width, 0, 
								x.size[0], self.height)) #top right
		img10 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(x.size[0] - self.width, x.size[1] - self.height, 
								x.size[0], x.size[1])) # bottom right
								
		return [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
		
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
		videoFullPath = os.path.join(self.dataDir, 'v', video_path)
		frames = os.listdir(videoFullPath)
		selectedFrames = []

		if self.is_training:
			selectedFrame = np.random.choice(frames[:-self.stack_length])
			frameId = frames.index(selectedFrame)
			frameStack = frames[frameId:frameId+self.stack_length]
			# self._train_stack_transform(frameStack, video_path)
			pathFrameStack = [[os.path.join(videoFullPath, f) for f in frameStack]]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0,len(frames[:-self.stack_length]),num=length,dtype=np.int32,endpoint=False)
			frameStack = [frames[frameId:frameId+self.stack_length]	for frameId in ids]
			# self._test_stack_transform(frameStack, video_path)
			pathFrameStack = [[os.path.join(videoFullPath, f) for f in fs] for fs in frameStack]

		return pathFrameStack

		
	def createMap(self, max_epochs, minibatch_size):
		mapFilesText = ['']*self.channel_count
		mapFilesPath = self.createMapPaths(len(mapFilesText))
		count = 0
		
		for epoch in range(max_epochs): # loop over epochs
			self.reset()
			while self.has_more():		# loop over minibatches in the epoch
				frames, labels, current_minibatch = self.next_minibatch(minibatch_size)
				for mb_fs, mb_l in zip(frames, labels):
					for fs in mb_fs:
						assert(len(mapFilesText) == 2*len(fs))
						for i, f in enumerate(mapFilesText):
							if i%2==0:
								mapFilesText[i] += '{}\t{}\t{}\n'.format(count, fs[int(i/2)], mb_l)
							else:
								mapFilesText[i] += '{}\t{}\t{}\n'.format(count, fs[int(i/2)].replace('\\v\\', '\\u\\'), mb_l)
						count+=1
					
			for i, f in enumerate(mapFilesText):
				with open(mapFilesPath[i], 'a') as file:
					file.write(f)
			mapFilesText = ['']*self.channel_count
			
			print('Finished epoch {}'.format(epoch))
			
	def createMapPaths(self, mapSize):
		if not os.path.exists(self.mapDir):
			os.mkdir(self.mapDir)
		mapFilesPath = []
		for i in range(mapSize):
			if self.is_training: 
				new_mapPath = os.path.join(self.mapDir, 'trainMap_{}.txt'.format(i+1))
			else:
				new_mapPath = os.path.join(self.mapDir, 'testMap_{}.txt'.format(i+1))
			mapFilesPath.append(new_mapPath)
			# Creating an empty file
			open(new_mapPath, 'w').close()
		return mapFilesPath

	def _train_stack_transform(self, frameStack, video_path):
		img = Image.open(os.path.join(self.dataDir, 'v', video_path, frameStack[0]))
		width  = img.size[0]
		height = img.size[1]
		startWidth	= (width - 224)*np.random.random_sample()
		startHeight = (height - 224)*np.random.random_sample()
		flipChance = np.random.random()
		# 10 frames in the stack
		for f in frameStack:
			img_u = Image.open(os.path.join(self.dataDir, 'u', video_path, f))
			img_v = Image.open(os.path.join(self.dataDir, 'v', video_path, f))
			u = img_u.crop((startWidth, startHeight, 
							startWidth+self.width, startHeight+self.height))
			v = img_v.crop((startWidth, startHeight, 
							startWidth+self.width, startHeight+self.height))
			if flipChance > 0.5:
				u = u.transpose(Image.FLIP_LEFT_RIGHT)
				v = v.transpose(Image.FLIP_LEFT_RIGHT)
			u.save(os.path.join(self.newImgDir, 'u', video_path, f))
			v.save(os.path.join(self.newImgDir, 'v', video_path, f))
	
	def _test_stack_transform(self, videoStack, video_path):
		# 25 for testing
		for frameStack in videoStack:
			seq_array = []
			# 10 frames in the stack
			for f in frameStack:
				img_u = Image.open(os.path.join(self.dataDir, 'u', video_path, f))
				img_v = Image.open(os.path.join(self.dataDir, 'v', video_path, f))
				seq_array.append([img_u, img_v])
			# Making 10 cropped stacks
			for cropType in self.multiView:
				for [u, v] in seq_array:
					u = cropType(u)
					v = cropType(v)
					u.save(os.path.join(self.newImgDir, 'u_test', video_path, f))
					v.save(os.path.join(self.newImgDir, 'v_test', video_path, f))
						
					
if __name__ == '__main__':
	dataDir		 = "F:/TCC/Datasets/UCF-101_opticalFlow"
	trainMapFile = os.path.join(dataDir, "trainlist01.txt")
	testMapFile	 = os.path.join(dataDir, "testlist01.txt")
	classMapFile = os.path.join(dataDir, "classInd.txt")
	newDirForMap = None #"/hdfs/pnrsy/t-pakova"
	mapDir		 = "F:/TCC/Datasets/OF_mapFiles-forLaterXforms"
	dirForNewImg = "F:/TCC/Datasets/UCF-101_opticalFlow_transformed"
	
	# Model dimensions
	stack_length = 10
	num_classes	 = 101

	max_epochs = 2147
	minibatch_size = 256
	
	# print('Creating train map files')
	# train_reader = createMapFile(trainMapFile, dataDir, 224, 224, stack_length, 
					# num_classes, max_epochs, minibatch_size, is_training=True, 
					# classMapFile=None, mapDir=mapDir, newImgDir=dirForNewImg)
	print('Creating test map files')
	test_reader = createMapFile(testMapFile, dataDir, 224, 224, stack_length, 
					num_classes, 1, minibatch_size, is_training=False, 
					classMapFile=classMapFile, mapDir=mapDir, newImgDir=dirForNewImg)

	
