from __future__ import print_function
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imsave
from cntk import load_model
from cntk import Trainer
from cntk.device import set_default_device, gpu
from cntk.ops import combine, input_variable, softmax
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms

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
		self.auxFile 		 = os.path.join(dataDir, 'auxFile.txt')

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
			video, videoPath = self._select_features(self.video_files[index])
			id = [i for i,v in enumerate(self.targets[index]) if v==1.0][0]
			with open(self.auxFile, 'a') as file:
				file.write(str(self.auxFile)+'\t'+str(id)+'\n')
			inputs[idx - self.batch_start, :, :, :, :] = video
			targets[idx - self.batch_start, :, :]		= self.targets[index]
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
		
		return video_frames, selectedFrames[0] #np.squeeze(video_frames)

	def _transform_frame(self, image_path):
		# load image
		img = Image.open(image_path)

		# Transformations
		img = img.resize((self.width, self.height), Image.ANTIALIAS)
			
		# Format image (RGB -> BGR, HWC -> CHW)
		bgr_image = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
		chw_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		image_data = chw_format - self.imageMean
		return image_data

	def resize(self, img, newWidth, newHeight):
		resized = img.resize((newWidth, newHeight), Image.ANTIALIAS)
		return resized
		

def create_reader(map_file, mean_file):
	transforms = [
		xforms.scale(width=224, height=224, channels=num_channels, interpolations='linear'),
		xforms.mean(mean_file)
	]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
		labels	 = StreamDef(field='label', shape=num_classes))))	# and second as 'label'


def load_mb_image(minibatch_source, epoch_size):
	# evaluate model and get desired node output
	features_si = minibatch_source['features']
	labels_si = minibatch_source['labels']
	sample_count = 0
	mb_videos = []
	while sample_count < epoch_size:
		mb = minibatch_source.next_minibatch(1)
		mb_videos.append(mb[features_si].asarray())
		sample_count +=1
	return mb_videos

def load_video_image(video_source, mb_size):
	while video_source.has_more():
		videos, labels, current_minibatch = video_source.next_minibatch(mb_size)
	return videos
	
def eval_img(loaded_model, image_data):
	output		 = loaded_model.eval({loaded_model.arguments[0]:image_data})
	predictions	 = softmax(np.squeeze(output)).eval()
	label		 = np.argmax(predictions)
	return label, predictions[label]*100

def save_img(imgData, imagePath):
	img = np.array(np.squeeze(imgData))
	img = np.rollaxis(np.rollaxis(img,1),2,1)
	img = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
	imsave(imagePath, img)
	
def readImgMean(image_path):
	# load and format image (RGB -> BGR, CHW -> HWC)
	img = Image.open(image_path)
	bgr_image = np.asarray(img, dtype=np.float32)[..., [2, 1, 0]]
	hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
	return hwc_format
	
	
if __name__ == '__main__':
	set_default_device(gpu(0))

	# define location of model and data and check existence
	base_folder = 'E:\TCC'
	models = ['ResNet_34_ucf101_rgb_py2']
	model_dir    = os.path.join(base_folder, "Models")
	dataDir      = os.path.join(base_folder, "DataSets",
	mean_file    = os.path.join(dataDir, "ImageNet1K_mean.xml")
	mean_imgPath = os.path.join(dataDir, "meanImg.jpg")
	map_file     = os.path.join(dataDir, "TestMapFiles_compare.txt")
	output_file  = os.path.join(base_folder, "comparingEvals-.txt")
	
	# create minibatch source
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_output_classes = 101

	with open(map_file, 'r') as file:
		lines = file.readlines()
		train_reader = VideoReader(map_file, dataDir, mean_imgPath, image_width, image_height, 
									num_channels, num_output_classes, is_training=True)
		
	for model_name in models:
		# load model
		model_file   = os.path.join(model_dir, model_name)
		loaded_model = load_model(model_file)
		if len(loaded_model.outputs)>1:
			loaded_model  = combine([loaded_model.outputs[2].owner])

		sum_diff = 0
		max_diff = 0
		total = 0
		equal = 0
		minibatch_source = create_mb_source(image_height, image_width, num_channels, 
												num_output_classes, mean_file, map_file)
		train_reader = VideoReader(map_file, mean_imgPath, image_width, image_height, 
									num_channels, num_output_classes, is_training=False)


		mb_data = load_mb_image(minibatch_source, len(lines))
		video_data = load_video_image(train_reader, len(lines))
		
		for (mb, video) in zip(mb_data,video_data):
			mb_label, mb_prec = eval_img(loaded_model, mb)
			video_label, video_prec = eval_img(loaded_model, video)
			diff = abs(mb_prec-video_prec)
			if diff > max_diff: max_diff = diff
			if mb_label==video_label:
				equal +=1
			sum_diff += diff
			total +=1
		with open(output_file, 'a') as file:
			file.write('{:<20} | '.format(model_name))
			file.write('{:<5} images |'.format(total))	
			file.write('{:<10}: {:^5.2f}%, {:<10}: {:^5.2f}% '.format('Mean diff', 
						sum_diff/total, 'Max diff', max_diff))
			file.write('{:<10}: {:^5.2f}%\n'.format('Equal class', float(equal)/total))
		
		
		
		
		