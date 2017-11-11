from cntk import load_model, Trainer, UnitType
from cntk.debugging import start_profiler, stop_profiler, enable_profiler
from cntk.device import gpu, try_set_default_device
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, softmax, sequence, combine, splice, reduce_mean
import itertools
import numpy as np
import os
import re
from PIL import Image
from models import *


# Paths
base_folder = "F:\TCC"
models_dir	= os.path.join(base_folder, "Models")
data_dir	= os.path.join(base_folder, "Datasets")

# Model dimensions
image_height  = 224
image_width	  = 224
stack_length  = 10
num_classes	  = 101


# Define the reader for both training and evaluation action.
class VideoReader(object):

	def __init__(self, map_file, dataDir, image_width, image_height, stack_length, 
					label_count, is_training=True, classMapFile=None):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.map_file		 = map_file
		self.label_count	 = label_count
		self.width			 = image_width
		self.height			 = image_height
		self.sequence_length = 250
		self.is_training	 = is_training
		self.multiView		 = self.getMultiView()
		self.stack_length	 = stack_length
		self.channel_count	 = 2*stack_length
		self.flowRange		 = 40.0
		self.imageRange		 = 255.0
		self.reescaleFlow	 = self.getFlowReescale()
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList		 = [None]*self.label_count

		if self.is_training:
			self.sequence_length = 1
		else:
			# Getting class id for test files
			self.classMap = dict()
			with open(classMapFile, 'r') as file:
				for line in file:
					[label, className] = line.replace('\n', '').split(' ')
					self.classMap[className] = label
		
		with open(map_file, 'r') as file:
			for row in file:
				if self.is_training:
					[video_path, label] = row.replace('\n','').split(' ')
				else:
					video_path, label = self.getTestClass(row)
				video_path = re.search('/(.*).avi', video_path).group(1)
				video_path = os.path.join(dataDir, 'v', video_path)
				self.video_files.append(video_path)
				target = [0.0] * self.label_count
				target[int(label)-1] = 1.0
				self.targets.append(target)
				if self.myAuxList[int(label)-1] == None:
					self.myAuxList[int(label)-1] = [len(self.targets)-1]
				else:
					self.myAuxList[int(label)-1].append(len(self.targets)-1)

		self.indices = np.arange(len(self.video_files))
		self.reset()

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
		if self.is_training:
			self.groupByTarget()
		self.batch_start = 0

	def groupByTarget(self):
		workList = self.myAuxList[::]
		if self.is_training:
			for x in workList:
				np.random.shuffle(x)
			np.random.shuffle(workList)
		grouped = list(itertools.zip_longest(*workList))
		np.random.shuffle(grouped)
		self.indices = [x for x in itertools.chain(*grouped) if x != None]

	def getFlowReescale(self):
		return lambda x: x*(self.flowRange/self.imageRange) - self.flowRange/2
		
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
								self.width, x.size[1])) # flip bottom left
		img9 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(x.size[0] - self.width, 0, 
								x.size[0], self.height)) # flip top right
		img10 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).crop(
								(x.size[0] - self.width, x.size[1] - self.height, 
								x.size[0], x.size[1])) # flip bottom right
								
		return [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
		
	def next_minibatch(self, batch_size):
		'''
		Return a mini batch of sequence frames and their corresponding ground truth.
		'''
		batch_end = min(self.batch_start + batch_size, self.size())
		current_batch_size = batch_end - self.batch_start
		
		if current_batch_size < 0:
			raise Exception('Reach the end of the training data.')

		inputs	= np.empty(shape=(current_batch_size, self.sequence_length, self.channel_count, self.height, self.width), dtype=np.float32)
		targets = np.empty(shape=(current_batch_size, self.sequence_length, self.label_count), dtype=np.float32)
		for idx in range(self.batch_start, batch_end):
			index = self.indices[idx]
			inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
			targets[idx - self.batch_start, :, :]	   = self.targets[index]
		self.batch_start += current_batch_size
		return inputs, targets, current_batch_size

	def _select_features(self, video_path):
		'''
		Select a sequence of frames from video_path and return them as a Tensor.
		'''
		frames = sorted(os.listdir(video_path), key=lambda x: int(re.findall(r'\d+',x)[0]))
		selectedFrames = []

		if self.is_training:
			selectedFrame = np.random.choice(frames[:-self.stack_length])
			frameId = frames.index(selectedFrame)
			frameStack = [[os.path.join(video_path, f) for f in frames[frameId:frameId+self.stack_length]]]
			video_frames = self._train_stack_transform(frameStack)
		else:
			length = self.sequence_length/10
			ids = np.linspace(0, len(frames[:-self.stack_length]), num=length, dtype=np.int32, endpoint=False)
			frameStack = [[os.path.join(video_path, f) for f in frames[frameId:frameId+self.stack_length]] 
							for frameId in ids]
			video_frames = self._test_stack_transform(frameStack)
			
		return video_frames

	def _train_stack_transform(self, videoStack):
		video_array = []
		# 1 for training
		for frameStack in videoStack:
			img = Image.open(frameStack[0])
			width  = img.size[0]
			height = img.size[1]
			startWidth	= (width - 224)*np.random.random_sample()
			startHeight = (height - 224)*np.random.random_sample()
			flipChance = np.random.random()
			stack_array = []
			# 10 frames in the stack
			for path_v in frameStack:
				path_u = path_v.replace('/v/', '/u/')
				img_u = Image.open(path_u)
				img_v = Image.open(path_v)
				u = img_u.crop((startWidth, startHeight, 
								startWidth+self.width, startHeight+self.height))
				v = img_v.crop((startWidth, startHeight, 
								startWidth+self.width, startHeight+self.height))
				if flipChance > 0.5:
					u = u.transpose(Image.FLIP_LEFT_RIGHT)
					v = v.transpose(Image.FLIP_LEFT_RIGHT)
				new_u, new_v = self.transformFlow(np.asarray(u, dtype=np.float32), 
													np.asarray(v, dtype=np.float32))
				stack_array.append(new_u)
				stack_array.append(new_v)
			video_array.append(stack_array)
			
		return video_array		
	
	def _test_stack_transform(self, videoStack):
		video_array = []
		# 25 for testing
		for frameStack in videoStack:
			seq_array = []
			# 10 frames in the stack
			for path_v in frameStack:				
				path_u = path_v.replace('/v/', '/u/')
				img_u = Image.open(path_u)
				img_v = Image.open(path_v)
				seq_array.append([img_u, img_v])
			
			# Making 10 cropped stacks
			for cropType in self.multiView:
				stack_array = []
				for [u, v] in seq_array:
					u = cropType(u)
					v = cropType(v)
					new_u, new_v = self.transformFlow(np.asarray(u, dtype=np.float32), 
													np.asarray(v, dtype=np.float32))
					stack_array.append(new_u)
					stack_array.append(new_v)
				video_array.append(stack_array)

		return video_array
	
	def transformFlow(self, u, v):
		# Reescale flow values
		u = self.reescaleFlow(u)
		v = self.reescaleFlow(v)
		# Get displacement field mean value
		# meanFlow = np.mean([u, v]) 
		# return (u - meanFlow), (v - meanFlow)
		return (u - np.mean(u)), (v - np.mean(v))
	

# Create a minibatch source.
def create_video_mb_source(map_files, num_channels, image_height, image_width, num_classes, max_epochs, 
							is_training=True):
	if is_training:
		transforms = [xforms.crop(crop_type='Center', crop_size=224)]
		randomize = True
	else:
		transforms = [xforms.crop(crop_type='MultiView10', crop_size=224)]
		randomize = False
	
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	transforms += [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
	
	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): StreamDef(field='image', transforms=transforms),
				   "label"+str(i): StreamDef(field='label', shape=num_classes)}
		sources.append(ImageDeserializer(map_file, StreamDefs(**streams)))

	return MinibatchSource(sources, max_sweeps=max_epochs, randomize=randomize)

def create_OF_model(num_classes, num_channels, image_height, image_width):
	inputs = []
	for c in range(num_channels):
		inputs.append(input_variable((1, image_height, image_width), name='input_{}'.format(c)))
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	input_reduceMean = [(i-reduce_mean(i, axis=[1,2])) for i in input_reescaleFlow]
	z = splice(*(i for i in input_reduceMean), axis=0, name='pre_input')
	
	label_var = input_variable(num_classes)
	features = {}
	for i in range(20):
		features['feature'+str(i)] = inputs[i]
	
	return dict({
		'model': z,
		'label': label_var}, 
		**features)
	
# Trains a transfer learning model
def train_model(train_reader, output_dir, log_file, train_mapFiles):
	# Learning parameters
	max_epochs		= 2147 # 9537 training videos on total
	minibatch_size	= 64

	# Image parameters
	image_height = train_reader.height
	image_width	 = train_reader.width
	num_channels = train_reader.channel_count
	num_classes	 = train_reader.label_count
	
	# Create fake network
	streamOF = create_OF_model(num_classes, num_channels, image_height, image_width)
	
	# Create train reader:
	reader = create_video_mb_source(train_mapFiles, 1, image_height, image_width, num_classes, max_epochs)
	
	input_map = {}
	for i in range(20):
		input_map[streamOF['feature'+str(i)]] = reader.streams["feature"+str(i)]

	# Start training
	for epoch in range(0, max_epochs):	 # loop over epochs
		train_reader.reset()
		while train_reader.has_more():			 # loop over minibatches in the epoch
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			mb = reader.next_minibatch(minibatch_size, input_map=input_map)
			stream_videos = streamOF['model'].eval(mb)
			print('Stream:')
			print('Mean: {:^6.5f}, std: {:^5.2f}, var: {:^5.2f}, max: {:^5.2f}, min: {:^5.2f}'.format(
				np.average(stream_videos), np.std(stream_videos), np.var(stream_videos), 
				np.amax(stream_videos), np.amin(stream_videos)))
			print('VideoMb:')
			print('Mean: {:^6.5f}, std: {:^5.2f}, var: {:^5.2f}, max: {:^5.2f}, min: {:^5.2f}'.format(
				np.average(videos), np.std(videos), np.var(videos),	np.amax(videos), np.amin(videos)))
			print(videos)
			print('---------------------------')

			
# Get the video label based on its frames evaluations
def getFinalLabel(predictedLabels, labelsConfidence):
	maxCount = max(predictedLabels.values())
	top_labels = [label for label in predictedLabels.keys() if predictedLabels[label]==maxCount]
	# Only one label, return it
	if (len(top_labels) == 1):
		confidence = labelsConfidence[top_labels[0]]/maxCount
	# 2 or more labels, need to check confidence
	else:
		topConfidence = dict()
		for label in top_labels:
			topConfidence[label] = labelsConfidence[label]/maxCount
		confidence = max(topConfidence.values())
		top_labels = [label for label in topConfidence.keys() if topConfidence[label]==confidence]
	return top_labels[0], confidence*100

	
# Evaluate network and writes output to file
def eval_and_write(loaded_model, test_reader, output_file):
	sample_count = 0
	results = ''
	with open(output_file, 'a') as file:
		while sample_count < test_reader.size():
			videos, labels, current_minibatch = test_reader.next_minibatch(1)
			sample_count += current_minibatch
			predictedLabels = dict((key, 0) for key in range(num_classes))
			labelsConfidence = dict((key, 0) for key in range(num_classes))
			correctLabel = [j for j,v in enumerate(labels[0][0]) if v==1.0][0]
			output = loaded_model.eval({loaded_model.arguments[0]:videos[0]})
			predictions = softmax(np.squeeze(output)).eval()
			top_classes = [np.argmax(p) for p in predictions]
			for i, c in enumerate(top_classes):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabel, label, confidence)
			if sample_count%50 == 0:
				file.write(results)
				results = ''
		file.write(results)


if __name__ == '__main__':
	try_set_default_device(gpu(0))

	#For training
	newModelName   = "VVG16_2_videoOF_part1"
	train_map_file = os.path.join(data_dir, "UCF-101_splits", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_opticalFlow")
	new_model_file = os.path.join(models_dir, newModelName)
	output_dir	   = os.path.join(base_folder, "Output-{}".format(newModelName))
	logFile		   = os.path.join(output_dir, "VGG16_log.txt")
	
	map_dir = os.path.join(data_dir, "UCF-101_ofMapFiles_split1")
	train_mapFiles = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f]
	
	train_reader = VideoReader(train_map_file, frames_dir, image_width, image_height, stack_length, 
								num_classes, is_training=True)
	train_model(train_reader, output_dir, logFile, train_mapFiles)
	
	