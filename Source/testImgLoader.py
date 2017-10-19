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

	def __init__(self, map_file, mean_file, image_width, image_height, num_channels, label_count, is_training):
		'''
		Load video file paths and their corresponding labels.
		'''
		self.map_file		 = map_file
		self.label_count	 = label_count
		self.width			 = image_width
		self.height			 = image_height
		self.sequence_length = 1
		self.channel_count	 = num_channels
		self.is_training	 = is_training
		self.video_files	 = []
		self.targets		 = []
		self.imageMean		 = self.readMean(mean_file)

		with open(map_file, 'r') as file:
			for row in file:
				[video_path, label] = row.replace('\n','').split('\t')
				self.video_files.append(video_path)
				target = [0.0] * self.label_count
				target[int(label)] = 1.0
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
		# if self.is_training:
			# np.random.shuffle(self.indices)
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
		video_frames = [self._transform_frame(video_path)]
		return video_frames

	def _transform_frame(self, image_path):
		# load image
		img = Image.open(image_path)
		if image_path.endswith("png"):
			temp = Image.new("RGB", img.size, (255, 255, 255))
			temp.paste(img, img)
			img = temp
		
		# Transformations
		if self.is_training:
			cropped = self.randomCrop(img)
			transformed = self.randomHFlip(cropped)
		else:
			transformed = self.resize(img)
			
		# Format image (RGB -> BGR, CHW -> HWC)
		bgr_image = np.asarray(transformed, dtype=np.float32)[..., [2, 1, 0]]
		hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
		# image_data = hwc_format - self.imageMean
		image_data = hwc_format
		return image_data

	def randomCrop(self, img):
		width = img.size[0]
		height = img.size[1]

		startWidth = int((width - self.width)*np.random.random_sample())
		startHeight = int((height - self.height)*np.random.random_sample())
		cropped = img.crop((startWidth, startHeight,
							startWidth+self.width, startHeight+self.height))
		return cropped
	
	def randomHFlip(self, img):
		chance = np.random.random()
		if chance >= 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		return img
		
	def resize(self, img):
		resized = img.resize((self.width, self.height), Image.ANTIALIAS)
		return resized

def create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, map_file):
	transforms = [
		xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
		# xforms.mean(mean_file)]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features=StreamDef(field='image', transforms=transforms),		# first column in map file is referred to as 'image'
		labels=StreamDef(field='label', shape=num_output_classes))),	# and second as 'label'.
		randomize=False)

def load_mb_image(minibatch_source, epoch_size):
	# evaluate model and get desired node output
	features_si = minibatch_source['features']
	labels_si = minibatch_source['labels']
	sample_count = 0
	mb_videos = []
	while sample_count < epoch_size:
		mb = minibatch_source.next_minibatch(1)
		mb_videos.append(mb[features_si].value)
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
	model_file	= os.path.join(base_folder, "Models", "resnet50_101output2")
	mean_file	= os.path.join(base_folder, "DataSets", "ImageNet1K_mean.xml")
	mean_imgPath	= os.path.join(base_folder, "Datasets", "meanImg.jpg")
	map_file = os.path.join(base_folder, "Datasets", "testLoadImg_map.txt")
	output_dir = os.path.join(base_folder, "Datasets", "TestLoadImg")

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	# create minibatch source
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_output_classes = 1000

	# load model
	loaded_model  = load_model(model_file)
	# loaded_model	= combine([loaded_model.outputs[2].owner])

	minibatch_source = create_mb_source(image_height, image_width, num_channels, 
										num_output_classes, mean_file, map_file)
	train_reader = VideoReader(map_file, mean_imgPath, image_width, image_height, 
									num_channels, num_output_classes, is_training=False)
	imageMean = readImgMean(mean_imgPath)
		
	with open(map_file, 'r') as file:
		lines = file.readlines()
		
	mb_data = load_mb_image(minibatch_source, len(lines))
	video_data = load_video_image(train_reader, len(lines))
	
	sum_diff = 0
	max_diff = 0
	total = 0
	for (mb, video) in zip(mb_data,video_data):
		# plt.figure(1)
		img = np.array(np.squeeze(mb))
		img = np.moveaxis(img,0, -1)
		img = np.asarray(img, dtype=np.uint8)[..., [2, 1, 0]]
		# plt.imshow(img)
		
		# plt.figure(2)
		img = np.array(np.squeeze(video))
		img = np.moveaxis(img,0, -1)
		img = np.asarray(img, dtype=np.uint8)[..., [2, 1, 0]]
		# plt.imshow(img)
		# plt.show()
		
		mb_label, mb_prec = eval_img(loaded_model, mb)
		video_label, video_prec = eval_img(loaded_model, video)
		print('{:<5}. {:^2} | {:^5.2f}%\n'.format('mb', mb_label, mb_prec))
		print('{:<5}. {:^2} | {:^5.2f}%\n'.format('video', video_label, video_prec))
		diff = abs(mb_prec-video_prec)
		if diff > max_diff: max_diff = diff
		if mb_label==video_label:
			print('{:<10} | {:^5.2f}%\n'.format('CORRECT', diff))
		else:
			print('{:<10} | {:^5.2f}%\n'.format('WRONG', diff))
		sum_diff += diff
		total +=1
		# print([np.mean(diff) for diff in mb-video])
		print('------------\n')
	print('Mean diff: {:^5.2f}%, Max diff: {:^5.2f}%'.format(sum_diff/total, max_diff))
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
