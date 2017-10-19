from cntk import load_model, Trainer, UnitType, Axis, Constant
from cntk.device import gpu, try_set_default_device
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.layers import Constant, Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import CloneMethod, combine, input_variable, placeholder, softmax, sequence
import itertools
import numpy as np
import os
from PIL import Image


# Paths
base_folder = "F:\TCC"
models_dir	= os.path.join(base_folder, "Models")
data_dir	= os.path.join(base_folder, "Datasets")

# Model dimensions
image_height = 224
image_width	 = 224
num_channels = 3
num_classes	 = 101


# Define the reader for both training and evaluation action.
class VideoReader(object):

	def __init__(self, map_file, dataDir, image_width, image_height, num_channels,
					label_count, is_training=True, classMapFile=None):
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
				video_path = os.path.join(dataDir, video_path)
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
		np.random.shuffle(list(itertools.izip_longest(*workList)))
		self.indices = [x for x in itertools.chain(*grouped) if x != None]
	
	def formatImg(self, img):
		# Scale
		img = img.resize((self.width,self.height), Image.ANTIALIAS)
		# Format image (RGB -> BGR, HWC -> CHW)
		bgr_image = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
		chw_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		return chw_format
		
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
		for idx in xrange(self.batch_start, batch_end):
			index = self.indices[idx]
			inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
			targets[idx - self.batch_start, :, :]	   = self.targets[index]
		self.batch_start += current_batch_size
		return inputs, targets, current_batch_size

	def _select_features(self, video_path):
		'''
		Select a sequence of frames from video_path and return them as a Tensor.
		'''
		frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('.')[0]))
		selectedFrames = []

		if self.is_training:
			selectedFrame = np.random.choice(frames)
			video_frames = [self._train_transform(os.path.join(video_path, selectedFrame))]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0, len(frames), num=length, dtype=np.int32, endpoint=False)
			selectedFrames = [os.path.join(video_path, frames[i]) for i in ids]
			video_frames = list(itertools.chain(*[self._test_transform(f) for f in selectedFrames]))
	
		return video_frames

	def _train_transform(self, image_path):
		# load image
		img = Image.open(image_path)
		if image_path.endswith("png"):
			temp = Image.new("RGB", img.size, (255, 255, 255))
			temp.paste(img, img)
			img = temp
		
		# Transformations
		w, h = self.getSizes(img, 256)					   # Get new size so the smallest side equals 256
		t1	= img.resize((w, h), Image.ANTIALIAS)		   # Upscale so the min size equals 256
		t2	= self.randomCrop(t1, 256, 256)				   # Crop random 256 square
		img = self.randomCrop(img, self.width, self.height) # Crop random 224 square
		# Random flip
		if np.random.random() > 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		img_array = self.formatImg(img)
		# Color transform
		randomValue = (0.4)*np.random.random_sample() - 0.2
		return np.array(img_array)*(1+randomValue)
		
	def _test_transform(self, image_path):
		# load image
		img = Image.open(image_path)
		if image_path.endswith("png"):
			temp = Image.new("RGB", img.size, (255, 255, 255))
			temp.paste(img, img)
			img = temp
		
		### MultiView10 ###
		# top left
		xOff1 = 0
		yOff1 = 0
		#top right
		xOff2 = img.size[0] - self.width
		yOff2 = 0
		# bottom left
		xOff3 = 0
		yOff3 = img.size[1] - self.height
		# bottom right
		xOff4 = img.size[0] - self.width
		yOff4 = img.size[1] - self.height
		# center
		xOff5 = (img.size[0] - self.width)/2
		yOff5 = (img.size[1] - self.height)/2
		
		flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
		img1 = img.crop((xOff1, yOff1, xOff1 + self.width, yOff1 + self.height)) # top left
		img2 = img.crop((xOff2, yOff2, xOff2 + self.width, yOff2 + self.height)) #top right
		img3 = img.crop((xOff3, yOff3, xOff3 + self.width, yOff3 + self.height)) # bottom left
		img4 = img.crop((xOff4, yOff4, xOff4 + self.width, yOff4 + self.height)) # bottom right
		img5 = img.crop((xOff5, yOff5, xOff5 + self.width, yOff5 + self.height)) # center
		img6 = flip_img.crop((xOff1, yOff1, xOff1 + self.width, yOff1 + self.height)) # flip top left
		img7 = flip_img.crop((xOff2, yOff2, xOff2 + self.width, yOff2 + self.height)) # flip top right
		img8 = flip_img.crop((xOff3, yOff3, xOff3 + self.width, yOff3 + self.height)) # flip bottom left
		img9 = flip_img.crop((xOff4, yOff4, xOff4 + self.width, yOff4 + self.height)) # flip bottom right
		img10 = flip_img.crop((xOff5, yOff5, xOff5 + self.width, yOff5 + self.height)) # flip center
							
		multiView = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
		
		return map(self.formatImg, multiView)

	def randomCrop(self, img, newWidth, newHeight):
		width = img.size[0]
		height = img.size[1]
		startWidth = (width - newWidth)*np.random.random_sample()
		startHeight = (height - newHeight)*np.random.random_sample()
		cropped = img.crop((startWidth, startHeight, 
							startWidth+newWidth, startHeight+newHeight))
		return cropped
	
	def getSizes(self, img, upScale):
		width = img.size[0]
		height = img.size[1]			
		x = upScale/min(width, height)
		if width<=height:
			return upScale, int(x*height)
		else:
			return int(x*width), upScale
	

# Create network model by updating last layer
def create_model(base_model, feature_node_name, last_hidden_node_name, num_classes, input_features):
	feature_node = find_by_name(base_model, feature_node_name)
	last_node	 = find_by_name(base_model, last_hidden_node_name)
	
	# Clone the desired layers
	cloned_layers = combine([last_node.owner]).clone(
		CloneMethod.freeze, {feature_node: placeholder(name='features')})
	
	# Add new dense layer for class prediction
	# feat_norm	 = input_features - Constant(114)
	feat_norm  = input_features
	cloned_out = cloned_layers(feat_norm)
	z		   = Dense(num_classes, activation=None, name='fc101') (cloned_out)
	return z
	
# Trains a transfer learning model
def train_model(network_path, train_reader, output_dir, log_file):
	# Learning parameters
	max_epochs = 537 # frames per each video | 9537 training videos on total
	minibatch_size = 256
	lr_per_mb = [0.01]*376 + [0.001]
	momentum_per_mb = 0.9
	l2_reg_weight = 0.0001
	
	# Image parameters
	image_height = train_reader.height
	image_width	 = train_reader.width
	num_channels = train_reader.channel_count
	num_classes	 = train_reader.label_count
	
	# Input variables
	input_var = input_variable((num_channels, image_height, image_width))
	label_var = input_variable(num_classes)
		
	# Create model
	base_model	= load_model(network_path)
	# z = create_model(base_model, 'features', 'z.x', num_classes, input_var)
	z = create_model(base_model, 'data', 'drop7', num_classes, input_var)
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape)) 
	
	# Loss and metric
	ce = cross_entropy_with_softmax(z, label_var)
	pe = classification_error(z, label_var)

	# Set learning parameters
	lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
	lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=train_reader.size(), 
											unit=UnitType.sample)
	mm_schedule = momentum_schedule(momentum_per_mb)

	# Progress writers
	progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs, 
						log_to_file=log_file, freq=10)]

	# Trainer object
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, unit_gain=False,
							l2_regularization_weight = l2_reg_weight)
	trainer = Trainer(z, (ce, pe), learner, progress_writers)

	# Restore training and get last sample_count and last_epoch
	last_trained_model = 'F:\TCC\Outputs\Output-VGG_videoRGB-part1\Models\VGG16_50_trainer.dnn'
	trainer.restore_from_checkpoint(last_trained_model)
	z = trainer.model
	
	with open(logFile, 'a') as file:
		file.write('\nVGG + Freeze + allTransforms + videoTestMinibatch')
		file.write('\nMinibatch_size = {}\n'.format(minibatch_size))
	
	sample_count = trainer.total_number_of_samples_seen
	last_epoch = sample_count/train_reader.size()
	print('Total number of samples seen: {} | Last epoch: {}\n'.format(sample_count, last_epoch))
	
	for epoch in xrange(last_epoch, max_epochs):	  # loop over epochs
		train_reader.reset()
		while train_reader.has_more():	  # loop over minibatches in the epoch
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			trainer.train_minibatch({input_var : videos, label_var : labels})
			sample_count += current_minibatch
		
		trainer.summarize_training_progress()
		percent = (float(sample_count)/(train_reader.size()*max_epochs))*100
		print ("Processed {} samples. {:^5.2f}% of total".format(sample_count, percent))
		if epoch%50 == 0:
			z.save(os.path.join(output_dir, 'Models', "VGG16_{}.model".format(epoch)))
			trainer.save_checkpoint(os.path.join(output_dir, 'Models', "VGG16_{}_trainer.dnn".format(epoch)))
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
	with open(output_file, 'a') as file:
		while sample_count < test_reader.size():
			videos_, labels_, current_minibatch = test_reader.next_minibatch(1)
			sample_count += current_minibatch
			predictedLabels = dict((key, 0) for key in xrange(num_classes))
			labelsConfidence = dict((key, 0) for key in xrange(num_classes))
			results = ''
			for labels, videos in zip(labels_, videos_):
				correctLabel = [j for j,v in enumerate(labels[0]) if v==1.0][0]
				for i, video in enumerate(videos):
					output = loaded_model.eval({loaded_model.arguments[0]:video})
					predictions = softmax(np.squeeze(output)).eval()
					top_class = np.argmax(predictions)
					predictedLabels[top_class] += 1
					labelsConfidence[top_class] += predictions[top_class]
				label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
				results += '{:^15} | {:^15} {:^15.2f}%\n'.format(correctLabel, label, confidence)
			file.write(results)

if __name__ == '__main__':
	try_set_default_device(gpu(0))

	#For training
	newModelName   = "VGG_videoRGB-part2-eval2"
	network_path   = os.path.join(models_dir, "VGG16_ImageNet_CNTK.model")
	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_rgb")
	new_model_file = os.path.join(models_dir, newModelName)
	output_dir	   = os.path.join(base_folder, "Output-{}".format(newModelName))
	logFile		   = os.path.join(output_dir, "ResNet34_log.txt")
	#For evaluation
	test_map_file  = os.path.join(data_dir, "ucfTrainTestlist", "testlist01.txt")
	class_map_file = os.path.join(data_dir, "ucfTrainTestlist", "classInd.txt")
	output_file	   = os.path.join(base_folder, "Results", "eval_{}.txt".format(newModelName))
	
	### Training ###
	# if not os.path.exists(output_dir):
		# os.mkdir(output_dir)
	
	# train_reader = VideoReader(train_map_file, frames_dir, image_width, image_height, num_channels, 
									# num_classes, is_training=True)
	# trained_model = train_model(network_path, train_reader, output_dir, logFile)
	
	# trained_model.save(new_model_file)
	# print("Stored trained model at %s" % new_model_file)
	
	test_model = os.path.join("F:\TCC\Models\VGG_videoRGB-part2")
	trained_model = load_model(test_model)
	## Evaluation ###
	if (os.path.exists(output_file)):
		raise Exception('The file {} already exist.'.format(output_file))

	with open(output_file, 'w') as results_file:
		results_file.write('{:<15} | {:<15}\n'.format('Correct label', 'Predicted label'))
	
	test_reader = VideoReader(test_map_file, frames_dir, image_width, image_height, num_channels,
							num_classes, is_training=False, classMapFile=class_map_file)
	# evaluate model and write out the desired output
	eval_and_write(trained_model, test_reader, output_file)
	
	print("Done. Wrote output to %s" % output_file)
	