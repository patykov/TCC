from cntk import load_model, Trainer, UnitType, Axis
from cntk.device import set_default_device, gpu, try_set_default_device
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.layers import Constant, Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import CloneMethod, combine, input_variable, placeholder, softmax, sequence
import itertools
import numpy as np
import os
from PIL import Image


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

	def __init__(self, map_file, dataDir, mean_file, image_width, image_height, num_channels, label_count, is_training):
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
		self.imageMean		 = self.readMean(mean_file)
		self.myAuxList       = [None]*self.label_count

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
			
	def has_more(self):
		if self.batch_start < self.size():
			return True
		return False

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
		frames = sorted(os.listdir(video_path))
		selectedFrames = []

		if self.is_training:
			selectedFrames = [np.random.choice(frames)]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0, len(frames), num=length, dtype=np.int32, endpoint=False)
			selectedFrames = [frames[i] for i in ids]
		
		selectedFrames = [os.path.join(video_path, f) for f in selectedFrames]
		video_frames = [self._transform_frame(f) for f in selectedFrames]
		
		return video_frames #np.squeeze(video_frames)

	def _transform_frame(self, image_path):
		# load image
		img = Image.open(image_path)
		if image_path.endswith("png"):
			temp = Image.new("RGB", img.size, (255, 255, 255))
			temp.paste(img, img)
			img = temp
		
		# Transformations
		if self.is_training:
			# w, h = self.getSizes(img, 256)                       # Get new size so the smallest side equals 256
			# t1  = img.resize((w, h), Image.ANTIALIAS)            # Upscale so the min size equals 256
			# t2  = self.randomCrop(t1, 256, 256)                  # Crop random 256 square
			t3  = self.randomCrop(img, self.width, self.height)   # Crop random 224 square
			img = self.randomHFlip(t3)                           # Random flip
		else:
			img = img.resize((self.width, self.height), Image.ANTIALIAS)
			
		# Format image (RGB -> BGR, HWC -> CHW)
		bgr_image = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
		chw_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		image_data = self.colorTransform(chw_format, 0.2) - self.imageMean
		image_data = chw_format - self.imageMean
		return image_data

	def randomCrop(self, img, newWidth, newHeight):
		width = img.size[0]
		height = img.size[1]
		startWidth = (width - newWidth)*np.random.random_sample()
		startHeight = (height - newHeight)*np.random.random_sample()
		cropped = img.crop((startWidth, startHeight, 
							startWidth+newWidth, startHeight+newHeight))
		return cropped
	
	def randomHFlip(self, img):
		chance = np.random.random()
		if chance > 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		return img

	def getSizes(self, img, upScale):
		width = img.size[0]
		height = img.size[1]			
		x = upScale/min(width, height)
		if width<=height:
			return upScale, int(x*height)
		else:
			return int(x*width), upScale
	
	def colorTransform(self, imageArray, value):
		randomValue = (2*value)*np.random.random_sample() - value
		return imageArray*randomValue
	
	
def find_arg_by_name(name, expression):
	vars = [i for i in expression.arguments if i.name == name]
	assert len(vars) == 1
	return vars[0]

# Create network model by updating last layer
def create_model(base_model, last_hidden_node_name, feature_node_name, num_classes, input_features):
	last_node    = find_by_name(base_model, last_hidden_node_name)
	# feature_node = find_by_name(base_model, feature_node_name)
	
	# Clone the desired layers
	cloned_layers = combine([last_node.owner]).clone(
		CloneMethod.clone, {input_features: placeholder(name='features')})
	
	# Add new dense layer for class prediction
	cloned_out = cloned_layers(input_features)
	z		   = Dense(num_classes, activation=None, name='fc101') (cloned_out)
	return z
	
# Trains a transfer learning model
def train_model(network_path, train_reader, output_dir, log_file):
	# Learning parameters
	max_epochs = 537 # frames per each video | 9537 training videos on total
	minibatch_size = 256
	lr_per_mb = [0.01]*100 + [0.001]
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
	z = create_model(base_model, 'z.x', "features", num_classes, input_var)
		
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
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, 
							l2_regularization_weight = l2_reg_weight)
	trainer = Trainer(z, (ce, pe), learner, progress_writers)

	with open(logFile, 'a') as file:
		file.write('\nMinibatch_size = {}\n'.format(minibatch_size))
	
	log_number_of_parameters(z) ; print()
	
	sample_count = 0
	for epoch in range(max_epochs):		  # loop over epochs
		train_reader.reset()
		while train_reader.has_more():	  # loop over minibatches in the epoch
			videos, labels, current_minibatch = train_reader.next_minibatch(minibatch_size)
			trainer.train_minibatch({input_var : videos, label_var : labels})
			sample_count += current_minibatch
		
		trainer.summarize_training_progress()
		percent = (sample_count/(train_reader.size()*max_epochs))*100
		print ("Processed {} samples. {:^5.2f}% of total".format(sample_count, percent))
		if epoch%50 == 0:
			z.save(os.path.join(output_dir, 'Models', "ResNet_34_{}.model".format(epoch)))

	return z

# Creates minibatch reader for evaluation
def create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, map_file):
	transforms = [
		xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
		xforms.crop(crop_type='MultiView10'),
		xforms.mean(mean_file)
	]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features=StreamDef(field='image', transforms=transforms),		# first column in map file is referred to as 'image'
		labels=StreamDef(field='label', shape=num_output_classes))),	# and second as 'label'.
		randomize=False)

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
	return top_labels[0], confidence

# Evaluate network and writes output to file
def eval_and_write(loaded_model, minibatch_source, action_id, epoch_size):
	# evaluate model and get desired node output
	features_si = minibatch_source['features']
	sample_count = 0
	predictedLabels = dict()
	labelsConfidence = dict()
	newResult = '{:^15} | '.format(action_id)
	while sample_count < epoch_size:
		mb = minibatch_source.next_minibatch(1)
		output = loaded_model.eval({loaded_model.arguments[0]:mb[features_si]})
		sample_count += mb[features_si].num_samples
		predictions = softmax(np.squeeze(output)).eval()
		top_class = np.argmax(predictions)
		if top_class in predictedLabels.keys():
			predictedLabels[top_class] += 1
			labelsConfidence[top_class] += predictions[top_class] * 100
		else:
			predictedLabels[top_class] = 1
			labelsConfidence[top_class] = predictions[top_class] * 100
	label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
	newResult += '{:^15} | {:^15.2f}%\n'.format(label, confidence)
	return newResult
	

if __name__ == '__main__':
	# set_default_device(gpu(0))
	try_set_default_device(gpu(0), acquire_device_lock=True)

	#For training
	newModelName   = "ResNet34_newModel_videoTrainer_537epochs"
	network_path   = os.path.join(models_dir, "ResNet34_ImageNet_CNTK.model")
	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	mean_img_path  = os.path.join(data_dir, "meanImg.jpg")
	frames_dir	   = os.path.join(data_dir, "UCF-101_rgb")
	new_model_file = os.path.join(models_dir, newModelName)
	output_dir	   = os.path.join(base_folder, "Output-{}".format(newModelName))
	logFile		   = os.path.join(output_dir, "ResNet34_log.txt")
	#For evaluation
	test_map_file  = os.path.join(data_dir, "TestMapFiles01_RGB")
	mean_file_path = os.path.join(data_dir, "ImageNet1K_mean.xml")
	output_file    = os.path.join(base_folder, "Results", "eval_{}.txt".format(newModelName))
	
	### Training ###
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	train_reader = VideoReader(train_map_file, frames_dir, mean_img_path, image_width, image_height, num_channels, 
									num_classes, is_training=True)
	trained_model = train_model(network_path, train_reader, output_dir, logFile)
	
	trained_model.save(new_model_file)
	print("Stored trained model at %s" % new_model_file)
	
	
	### Evaluation ###
	if (os.path.exists(output_file)):
		raise Exception('The file {} already exist.'.format(output_file))

	#Get all test map files
	map_files = sorted(os.listdir(test_map_file))
	with open(output_file, 'a') as results_file:
		results_file.write('{:<15} | {:<15} | {:<15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	myResults = []
	for test_file in map_files:
		action_id = test_file.split('_')[-1][:-4]
		file_path = os.path.join(test_map_file, test_file)
		minibatch_source = create_mb_source(image_height, image_width, num_channels, num_classes, 
												mean_file_path, file_path)
		# evaluate model and write out the desired output
		result = eval_and_write(trained_model, minibatch_source, action_id, epoch_size=250)	  # 25 frames for that result in 250 inputs for the network
		myResults.append(result)
		if len(myResults) >= 100:
			with open(output_file, 'a') as results_file:
				for result in myResults:
					results_file.write(result)
			myResults = []
		
	# Saving the myResults < 100 left
	with open(output_file, 'a') as results_file:
		for result in myResults:
			results_file.write(result)

	print("Done. Wrote output to %s" % output_file)

