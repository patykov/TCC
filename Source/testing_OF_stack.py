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
num_channels = 2	# u, v
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
		self.stack_length	 = 10
		self.channel_count	 = num_channels
		self.is_training	 = is_training
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList       = [None]*self.label_count

		with open(map_file, 'r') as file:
			for row in file:
				[video_path, label] = row.replace('\n','').split(' ')
				video_path = os.path.join(dataDir, 'v', video_path)
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
			t1  = self.randomCrop(img, self.width, self.height)  # Crop random 224 square
			img = self.randomHFlip(t1)                           # Random flip
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


if __name__ == '__main__':

	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_opticalFlow")
	stack = 10
	video_path = 'E:/TCC/Datasets/UCF-101_opticalFlow/v/v_ApplyEyeMakeup_g01_c05'
	
	frames = sorted(os.listdir(video_path))
	selectedFrame = np.random.choice(frames)
	print(selectedFrame)
	frameId = frames.index(selectedFrame)
	selectedFrames = [f for f in frames[frameId:frameId+stack]]

	img = Image.open(os.path.join(video_path,selectedFrame))
	width = img.size[0]
	height = img.size[1]
	startWidth = (width - 224)*np.random.random_sample()
	startHeight = (height - 224)*np.random.random_sample()
	chance = np.random.random()
	img_array = []
	for f in selectedFrames:
		path_v = os.path.join(video_path, f)
		path_u = path_v.replace('/v/', '/u/')
	
		img_v = Image.open(path_v)
		img_u = Image.open(path_u)
		
		cropped_v = img_v.crop((startWidth, startHeight, startWidth+224, startHeight+224))
		cropped_u = img_u.crop((startWidth, startHeight, startWidth+224, startHeight+224))
		if chance > 0.5:
			cropped_v = cropped_v.transpose(Image.FLIP_LEFT_RIGHT)
			cropped_u = cropped_u.transpose(Image.FLIP_LEFT_RIGHT)
		img_array.append(np.asarray(cropped_u, dtype=np.float32))
		img_array.append(np.asarray(cropped_v, dtype=np.float32))
	print(np.array(img_array).shape)
	
	print (img_array[0])
	
	newFlow = img_array[0]*(40.0/255.0) - 40.0/2
	print (newFlow)
	print(np.mean(newFlow))
	print(newFlow-np.mean(newFlow))
	
	
	
	
	
	