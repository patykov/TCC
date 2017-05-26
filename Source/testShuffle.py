import numpy as np
import os
import itertools


# Paths
base_folder = "E:\TCC"
models_dir	= os.path.join(base_folder, "Models")
data_dir	= os.path.join(base_folder, "Datasets")

# Model dimensions
image_height = 224
image_width	 = 224
num_channels = 3
num_classes	 = 101

# Define the reader for both training and evaluation action.
class VideoReader(object):

	def __init__(self, map_file, dataDir, image_width, image_height, num_channels, label_count, is_training):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.map_file		 = map_file
		self.label_count	 = label_count
		self.width			 = image_width
		self.height			 = image_height
		self.sequence_length = 250
		self.channel_count	 = num_channels
		self.is_training	 = is_training
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList = [None]*self.label_count


		with open(map_file, 'r') as file:
			for row in file:
				[video_path, label] = row.replace('\n','').split(' ')
				video_path = os.path.join(dataDir, video_path)
				self.video_files.append(video_path)
				target = [0.0] * self.label_count
				target[int(label)-1] = 1.0
				self.targets.append(target)
				if self.myAuxList[int(label)-1] == None:
					self.myAuxList[int(label)-1] = [len(self.targets)-1]
				else:
					self.myAuxList[int(label)-1].append(len(self.targets)-1)

		if self.is_training:
			self.sequence_length = 1
		self.indices = np.arange(len(self.video_files))
		self.reset()

	def size(self):
		return len(self.video_files)
			
	def reset(self):
		self.groupByTarget()
		self.batch_start = 0

	def groupByTarget(self):
		workList = self.myAuxList[::]
		if self.is_training:
			for x in workList:
				np.random.shuffle(x)
		workList.sort(key=len, reverse=True)
		aux = list(itertools.zip_longest(*workList))
		self.indices = [x for x in itertools.chain(*list(itertools.zip_longest(*workList))) if x != None]
		

if __name__ == '__main__':
	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_rgb")

	
	train_reader = VideoReader(train_map_file, frames_dir, image_width, image_height, num_channels, 
									num_classes, is_training=True)

