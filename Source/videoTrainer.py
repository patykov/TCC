import numpy as np
import os
from PIL import Image
from cntk import load_model, Trainer, UnitType, Axis
from cntk.device import set_default_device, gpu
from cntk.io import MinibatchSource, ImageDeserializer
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, combine, softmax, sequence
from cntk.utils import ProgressPrinter


################################################
################################################
# general settings
base_folder = "E:\TCC"
outputDir = os.path.join(base_folder, "Output-learnrAndLrRate")
modelsDir = os.path.join(outputDir, "Models")
logFile = os.path.join(outputDir, "ResNet34_log.txt")
new_model_file = os.path.join(outputDir, "ResNet34_UCF101_videoReader")
output_file = os.path.join(outputDir, "predOutput.txt")

# define base model location and characteristics
_base_model_file = os.path.join(base_folder, "Models", "ResNet_34_101output")
_feature_node_name = "features"
_last_hidden_node_name = "pool5"
_image_height = 224
_image_width = 224
_num_channels = 3

# define data location and characteristics
_data_folder = os.path.join(base_folder, "DataSets")
_framesDir = os.path.join(_data_folder, "UCF-101_rgb")
_mean_file = os.path.join(_data_folder, "meanImg.jpg")
_train_map_file = os.path.join(_data_folder, "ucfTrainTestlist", "trainlist01.txt")
_num_classes = 101
################################################
################################################


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

		with open(map_file, 'r') as file:
			for row in file:
				[video_path, label] = row.replace('\n','').split(' ')
				video_path = os.path.join(dataDir, video_path)
				self.video_files.append(video_path)
				target = [0.0] * self.label_count
				target[int(label)-1] = 1.0
				self.targets.append(target)

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
		if self.is_training:
			np.random.shuffle(self.indices)
		self.indices = self.groupByTarget(self.indices)
		self.batch_start = 0

	def groupByTarget(self, indices):
		newInd = []
		myTargets = []
		usedIndices = []
		while len(newInd) < len(self.video_files):
			for j in indices:
				if self.targets[j] not in myTargets:
					newInd.append(j)
					usedIndices.append(j)
					myTargets.append(self.targets[j])
				if (len(myTargets) >= self.label_count) or (len(newInd) >= len(self.video_files)):
					indices = np.delete(indices, usedIndices)
					myTargets = []
					usedIndices = []
					break
		return np.array(newInd)
		
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
		video_frames = [self._transform_frame(f) for f in selectedFrames]
		
		return video_frames		# return np.stack(video_frames, axis=0)

	def _transform_frame(self, image_path):
		# load image
		img = Image.open(image_path)
		if image_path.endswith("png"):
			temp = Image.new("RGB", img.size, (255, 255, 255))
			temp.paste(img, img)
			img = temp
		
		# Transformations
		if self.is_training:
			transformed = self.randomCrop(img)
		else:
			transformed = self.resize(img)
			
		# Format image (RGB -> BGR, CHW -> HWC)
		bgr_image = np.asarray(transformed, dtype=np.float32)[..., [2, 1, 0]]
		hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		image_data = hwc_format - self.imageMean
		return image_data

	def randomCrop(self, img):
		width = img.size[0]
		height = img.size[1]

		startWidth = (width - self.width)*np.random.random_sample()
		startHeight = (height - self.height)*np.random.random_sample()
		cropped = img.crop((startWidth, startHeight,
							startWidth+self.width, startHeight+self.height))
		return cropped
		
	def rezise(self, img):
		resized = img.resize((self.width, self.height), Image.ANTIALIAS)
		return resized

def find_arg_by_name(name, expression):
	vars = [i for i in expression.arguments if i.name == name]
	assert len(vars) == 1
	return vars[0]

# Trains a transfer learning model
def train_model(base_model_file, mean_file, image_width, image_height, num_channels, 
				num_classes, train_map_file, framesDir):
	# Learning parameters
	max_epochs = 150 # frames per each video | 9537 training videos on total
	mb_size = 101
	lr_per_mb = [0.004]*100 + [0.0004]
	momentum_per_mb = 0.9
	l2_reg_weight = 0.0001
		
	loaded_model = load_model(base_model_file)
	loaded_model = combine([loaded_model.outputs[2].owner])
		
	# Create the minibatch source and input variables
	train_reader = VideoReader(train_map_file, framesDir, mean_file, 
									image_width, image_height, num_channels, 
									num_classes, is_training=True)
	
	image_input = find_arg_by_name('features',loaded_model)
	label_input = input_variable(num_classes, 
								dynamic_axes=loaded_model.dynamic_axes,
								name='labels')
	
	ce = cross_entropy_with_softmax(loaded_model, label_input)
	pe = classification_error(loaded_model, label_input)
	
	# Instantiate the trainer object
	lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch, epoch_size=train_reader.size())
	mm_schedule = momentum_schedule(momentum_per_mb)
	learner = momentum_sgd(loaded_model.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
	progress_printer = ProgressPrinter(tag='Training', freq=10, num_epochs=max_epochs, log_to_file=logFile)
	trainer = Trainer(loaded_model, (ce, pe), learner, progress_printer)

	# Get minibatches of images to train with and perform model training
	sample_count = 0
	for epoch in range(max_epochs):		  # loop over epochs
		train_reader.reset()
		while train_reader.has_more():
			videos, labels, current_minibatch = train_reader.next_minibatch(mb_size)
			trainer.train_minibatch({image_input : videos, label_input : labels})
			sample_count += current_minibatch
		trainer.summarize_training_progress()
		percent = (sample_count/(train_reader.size()*max_epochs))*100
		print ("Processed {} samples. {:^5.2f}% of total".format(sample_count, percent))
		if epoch%10 == 0:
			loaded_model.save_model(os.path.join(modelsDir, "ResNet_34_{}.model".format(epoch)))


	return loaded_model


if __name__ == '__main__':
	set_default_device(gpu(0))
	
	if not os.path.exists(outputDir):
		os.mkdir(outputDir)
	
	trained_model = train_model(_base_model_file, _mean_file, _image_width, _image_height, 
								_num_channels, _num_classes, _train_map_file, _framesDir)
	trained_model.save_model(new_model_file)
	print("Stored trained model at %s" % new_model_file)


