from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint

from PIL import Image
# import imageio

from cntk import Trainer
from cntk import load_model
from cntk.utils import *
from cntk.layers import *
from cntk.learner import sgd, momentum_sgd, learning_rate_schedule, momentum_schedule, momentum_as_time_constant_schedule, UnitType
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, relu, minus, element_times, constant, softmax
from _cntk_py import set_computation_network_trace_level

# Paths relative to current python file.
# abs_path	 = os.path.dirname(os.path.abspath(__file__))
abs_path = "E:\TCC"
data_path  = os.path.join(abs_path, "Datasets")
model_path = os.path.join(abs_path, "Models", "ResNet34_UCF101_rgb_py2")

# Define the reader for both training and evaluation action.
class VideoReader(object):
	'''
	A simple VideoReader: 
	It iterates through each video and select 16 frames as
	stacked numpy arrays.
	Similar to http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
	'''
	def __init__(self, map_file, dataDir, mean_file, label_count, is_training):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.map_file		 = map_file
		self.label_count	 = label_count
		self.width			 = 224
		self.height			 = 224
		self.sequence_length = 250
		self.channel_count	 = 3
		self.is_training	 = is_training
		self.video_files	 = []
		self.targets		 = []
		self.batch_start	 = 0
		self.imageMean 		 = self.readMean(mean_file)

		with open(map_file, 'r') as file:
			for row in file:
				[video_path, label] = row.replace('\n','').split(' ')
				video_path = os.path.join(dataDir, video_path)
				self.video_files.append(video_path)
				target = [0.0] * self.label_count
				# In the map file the labels are [1,101], cntk need it [0,100]
				target[int(label)-1] = 1.0
				self.targets.append(target)

		self.indices = np.arange(len(self.video_files))
		if self.is_training:
			np.random.shuffle(self.indices)
			self.sequence_length = 1

	def size(self):
		return len(self.video_files)
			
	def has_more(self):
		if self.batch_start < self.size():
			return True
		return False

	def reset(self):
		if self.is_training:
			np.random.shuffle(self.indices)
		self.batch_start = 0

	def readMean(self, image_path):
		# load and format image (RGB -> BGR, CHW -> HWC)
		img = Image.open(image_path)
		bgr_image = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
		hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		return hwc_format
		
	def next_minibatch(self, batch_size):
		'''
		Return a mini batch of sequence frames and their corresponding ground truth.
		'''
		batch_end = min(self.batch_start + batch_size, self.size())
		current_batch_size = batch_end - self.batch_start
		if current_batch_size < 0:
			raise Exception('Reach the end of the training data.')

		inputs	= np.empty(shape=(current_batch_size, self.channel_count, self.sequence_length, self.height, self.width), dtype=np.float32)
		targets = np.empty(shape=(current_batch_size, self.label_count), dtype=np.float32)
		for idx in range(self.batch_start, batch_end):
			index = self.indices[idx]
			inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
			targets[idx - self.batch_start, :]		   = self.targets[index]

		self.batch_start += current_batch_size
		return inputs, targets, current_batch_size

	def _select_features(self, video_path):
		'''
		Select a sequence of frames from video_path and return them as
		a Tensor.
		'''
		frames = sorted(os.listdir(video_path))
		selectedFrames = []

		if self.is_training:
			selectedFrames = [np.random.choice(frames)]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0, len(frames), num=length, dtype=np.int32, endpoint=False)
			selectedFrames = [frames[i] for i in ids]

		selectedFrames = [os.path.join(video_path, f) for f in selectedFrames]
		print(selectedFrames)
		video_frames = [self._transform_frame(f) for f in selectedFrames]
		
		return np.stack(video_frames, axis=1)

	def _transform_frame(self, image_path):
		'''
		For training and testing we select a 224x224 random crop 
		and reduce a mean value. For testing we also produce a 
		10-view image evaluation: 
		(4(corners) + 1(center)) * 2(original + horizontal flip) = 10.
		'''
		if not self.is_training:
			"TODO"
			image = Image.fromarray(data)
			width = image.size[0]
			height = image.size[1]

			startWidth = (width - 224)*np.random.random_sample()
			startHeight = (height - 224)*np.random.random_sample()

			image = image.crop((startWidth,
								startHeight,
								startWidth+224,
								startHeight+224))
			
			norm_image = np.array(image, dtype=np.float32)
			norm_image -= 128.0
			norm_image /= 128.0

			# (channel, height, width)
			return np.ascontiguousarray(np.transpose(norm_image, (2, 0, 1)))
			
		else:
			# load and format image (resize, RGB -> BGR, CHW -> HWC)
			img = Image.open(image_path)
			if image_path.endswith("png"):
				temp = Image.new("RGB", img.size, (255, 255, 255))
				temp.paste(img, img)
				img = temp
			# resized = img.resize((224, 224), Image.ANTIALIAS)
			width = img.size[0]
			height = img.size[1]

			startWidth = int((width - 224)*np.random.random_sample())
			startHeight = int((height - 224)*np.random.random_sample())
			resized = img.crop((startWidth,
								startHeight,
								startWidth+224,
								startHeight+224))
								
			bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
			hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
			image_data = hwc_format - self.imageMean
			return image_data


# Creates and trains a feedforward classification model for UCF11 action videos
def conv3d_ucf11(train_reader, max_epochs=30):
	# Replace 0 with 1 to get detailed log.
	set_computation_network_trace_level(0)

	# These values must match for both train and test reader.
	image_height	   = train_reader.height
	image_width		   = train_reader.width
	num_channels	   = train_reader.channel_count
	sequence_length	   = train_reader.sequence_length
	num_output_classes = train_reader.label_count

	# Input variables denoting the features and label data
	input_var = input_variable((num_channels, sequence_length, image_height, image_width), np.float32)
	label_var = input_variable(num_output_classes, np.float32)

	# Instantiate simple 3D Convolution network inspired by VGG network 
	# and http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf
	with default_options (activation=relu):
		z = Sequential([
			Convolution((3,3,3), 64, pad=True),
			MaxPooling((1,2,2), (1,2,2)),
			LayerStack(3, lambda i: [
				Convolution((3,3,3), [96, 128, 128][i], pad=True),
				Convolution((3,3,3), [96, 128, 128][i], pad=True),
				MaxPooling((2,2,2), (2,2,2))
			]),
			LayerStack(2, lambda : [
				Dense(1024), 
				Dropout(0.5)
			]),
			Dense(num_output_classes, activation=None)
		])(input_var)
	
	# loss and classification error.
	ce = cross_entropy_with_softmax(z, label_var)
	pe = classification_error(z, label_var)

	# training config
	epoch_size	   = 1322				   # for now we manually specify epoch size
	minibatch_size = 4

	# Set learning parameters
	lr_per_sample		   = [0.01]*10+[0.001]*10+[0.0001]
	lr_schedule			   = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
	momentum_time_constant = 4096
	mm_schedule			   = momentum_as_time_constant_schedule(momentum_time_constant, epoch_size=epoch_size)

	# Instantiate the trainer object to drive the model training
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, True)
	progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
	trainer = Trainer(z, (ce, pe), learner, progress_printer)

	log_number_of_parameters(z) ; print()

	# Get minibatches of images to train with and perform model training
	for epoch in range(max_epochs):		  # loop over epochs
		train_reader.reset()

		while train_reader.has_more():
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			trainer.train_minibatch({input_var : videos, label_var : labels})

		trainer.summarize_training_progress()
	
	# # Test data for trained model
	# epoch_size	 = 332
	# minibatch_size = 2

	# # process minibatches and evaluate the model
	# metric_numer	  = 0
	# metric_denom	  = 0
	# minibatch_index = 0

	# test_reader.reset()	 
	# while test_reader.has_more():
	#	  videos, labels, current_minibatch = test_reader.next_minibatch(minibatch_size)
	#	  # minibatch data to be trained with
	#	  metric_numer += trainer.test_minibatch({input_var : videos, label_var : labels}) * current_minibatch
	#	  metric_denom += current_minibatch
	#	  # Keep track of the number of samples processed so far.
	#	  minibatch_index += 1

	# print("")
	# print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
	# print("")

	return metric_numer/metric_denom

if __name__=='__main__':
	num_output_classes = 101
	minibatch_size = 2

	loaded_model = load_model(model_path)

	train_reader = VideoReader(os.path.join(data_path, 'train_map_video.txt'), 
								os.path.join(data_path, 'UCF-101_rgb'), 
								os.path.join(data_path, 'meanImg.jpg'),
								num_output_classes, True)

	while train_reader.has_more():
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			for image_data, image_label in zip(videos, labels):
				output		 = loaded_model.eval({loaded_model.arguments[0]:[np.squeeze(image_data)]})
				predictions	 = softmax(np.squeeze(output)).eval()
				label		 = np.argmax(predictions)
				correct_label = [i for i,x in enumerate(image_label) if int(x) == 1][0]
				print('{:^2} | {:^2} | {:^5.2f}%\n'.format(correct_label, label, predictions[label]*100))


	# trainModel(loaded_model, train_reader)
	
	# test_reader  = VideoReader(os.path.join(data_path, 'test_map.csv'), num_output_classes, False)	
	# conv3d_ucf11(train_reader)
