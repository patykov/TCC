from cntk import load_model, Trainer, UnitType
from cntk.debugging import start_profiler, stop_profiler, enable_profiler
from cntk.device import gpu, try_set_default_device
from cntk.io import StreamDef
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, softmax, sequence, combine, CloneMethod, placeholder
import itertools
import numpy as np
import os
import re
from PIL import Image
from models import *


# Paths
base_folder = "E:\TCC"
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
	

# Trains a transfer learning model
def train_model(train_reader, output_dir, log_file):
	# Learning parameters
	max_epochs		= 2147 # 9537 training videos on total
	minibatch_size	= 64
	lr_per_mb		= [0.01]*1341 + [0.001]*538 + [0.0001]
	momentum_per_mb = 0.9
	l2_reg_weight	= 0.0001

	# Image parameters
	image_height = train_reader.height
	image_width	 = train_reader.width
	num_channels = train_reader.channel_count
	num_classes	 = train_reader.label_count
	
	# Input variables
	input_var = input_variable((num_channels, image_height, image_width))
	label_var = input_variable(num_classes)

	# create model
	z = create_vgg16_2(input_var, num_classes)
	# z = load_model('F:\TCC\Output-ResNet34_videoOF\Models\ResNet_34_800_trainer.dnn')
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))
	# for index in range(len(z.outputs)):
		# print("Index {} for output: {} | {}.".format(index, z.outputs[index].name, z.outputs[index].shape))
		
	# Loss and metric
	ce = cross_entropy_with_softmax(z, label_var)
	pe = classification_error(z, label_var)

	# Set learning parameters
	lr_per_sample = [lr/256 for lr in lr_per_mb]
	lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=train_reader.size(), 
											unit=UnitType.sample)
	mm_schedule = momentum_schedule(momentum_per_mb)

	# Printer
	progress_printer = ProgressPrinter(freq=10, tag='Training', log_to_file=log_file, num_epochs=max_epochs)
	with open(logFile, 'a') as file:
		file.write('\nMinibatch_size = {}\n'.format(minibatch_size))

	# Trainer object
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, 
							l2_regularization_weight = l2_reg_weight)
	trainer = Trainer(z, (ce, pe), learner, progress_printer)

	# Restore training and get last sample_count and last_epoch
	last_trained_model = 'F:/TCC/Output-VVG16_2_videoOF_part1/Models/VGG16_140_trainer.dnn'
	trainer.restore_from_checkpoint(last_trained_model)
	z = trainer.model

	sample_count = trainer.total_number_of_samples_seen
	last_epoch = int(sample_count/train_reader.size())
	print('Total number of samples seen: {} | Last epoch: {}\n'.format(sample_count, last_epoch))
	
	# Start training
	start_profiler()
	for epoch in range(last_epoch, max_epochs):	 # loop over epochs
		train_reader.reset()
		while train_reader.has_more():			 # loop over minibatches in the epoch
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			trainer.train_minibatch({input_var : videos, label_var : labels})
			sample_count += current_minibatch
		trainer.summarize_training_progress()
		enable_profiler() # begin to collect profiler data after first epoch
		
		# Save checkpoint and model		
		percent = (float(sample_count)/(train_reader.size()*max_epochs))*100
		print ("Processed {} samples. {:^5.2f}% of total".format(sample_count, percent))
		if epoch%10 == 0:
			z.save(os.path.join(output_dir, 'Models', "VGG16_{}.model".format(epoch)))
			trainer.save_checkpoint(os.path.join(output_dir, 'Models', "VGG16_{}_trainer.dnn".format(epoch)))
	stop_profiler()
	
	return z
			
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
	newModelName   = "VGG16_videoOF_unitT"
	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_opticalFlow")
	new_model_file = os.path.join(models_dir, newModelName)
	output_dir	   = os.path.join(base_folder, "Output-{}".format(newModelName))
	logFile		   = os.path.join(output_dir, "VGG16_log.txt")
	#For evaluation
	test_map_file  = os.path.join(data_dir, "UCF-101_splits", "testlist01.txt")
	class_map_file = os.path.join(data_dir, "UCF-101_splits", "classInd.txt")
	output_file	   = os.path.join(base_folder, "Results", "eval_{}.txt".format(newModelName))
	
	### Training ###
	# if not os.path.exists(output_dir):
		# os.mkdir(output_dir)
	
	# train_reader = VideoReader(train_map_file, frames_dir, image_width, image_height, stack_length, 
								# num_classes, is_training=True)
	# trained_model = train_model(train_reader, output_dir, logFile)
	
	# trained_model.save(new_model_file)
	# print("Stored trained model at %s" % new_model_file)
	
	test_model = "E:/TCC/Models/philly/VGG16_videoOF_unitT.dnn"
	trained_model = load_model(test_model)
	
	## Evaluation ###
	if (os.path.exists(output_file)):
		raise Exception('The file {} already exist.'.format(output_file))

	with open(output_file, 'w') as results_file:
		results_file.write('{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	test_reader = VideoReader(test_map_file, frames_dir, image_width, image_height, stack_length, 
								num_classes, is_training=False, classMapFile=class_map_file)
	# evaluate model and write out the desired output
	eval_and_write(trained_model, test_reader, output_file)
	
	print("Done. Wrote output to %s" % output_file)