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

def multiView():
	img2 = lambda x: x.transpose(Image.ROTATE_90)
	img2.__name__ = '90'
	img3 = lambda x: x.transpose(Image.ROTATE_180)
	img3.__name__ = '180'
	img4 = lambda x: x.transpose(Image.ROTATE_270)
	img5 = lambda x: x.transpose(Image.TRANSPOSE)
	img6 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)
	img7 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
	img8 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
	img9 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
	img10 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.TRANSPOSE)
	return [img2, img3, img4, img5, img6, img7, img8, img9, img10]

if __name__ == '__main__':

	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_opticalFlow")
	stack = 10
	video_path = 'E:/TCC/Datasets/UCF-101_opticalFlow/v/v_ApplyEyeMakeup_g01_c05'
	multiView = multiView()
	
	frames = sorted(os.listdir(video_path))
	selectedFrames = []
	is_training = True
	if is_training:
		selectedFrame = np.random.choice(frames[:-stack])
		frameId = frames.index(selectedFrame)
		videoStack = [[os.path.join(video_path, f) for f in frames[frameId:frameId+stack]]]
	else:
		length = 250/10
		ids = np.linspace(0, len(frames[:-stack]), num=length, dtype=np.int32, endpoint=False)
		videoStack = [[os.path.join(video_path, f) for f in frames[frameId:frameId+stack]] 
						for frameId in ids]
	print(len(videoStack))
	print(len(videoStack[0]))
	img = Image.open(videoStack[0][0])
	width = img.size[0]
	height = img.size[1]
	video_array = []
	for imgStack in videoStack:
		startWidth = (width - 224)*np.random.random_sample()
		startHeight = (height - 224)*np.random.random_sample()
		chance = np.random.random()
		img_array = []
		stack_array = []
		for path_v in imgStack:
			path_u = path_v.replace('/v/', '/u/')
			img_u = Image.open(path_u)
			img_v = Image.open(path_v)
			
			cropped_u = img_u.crop((startWidth, startHeight, startWidth+224, startHeight+224))
			cropped_v = img_v.crop((startWidth, startHeight, startWidth+224, startHeight+224))
			if is_training:
				if chance > 0.5:
					cropped_u = cropped_u.transpose(Image.FLIP_LEFT_RIGHT)
					cropped_v = cropped_v.transpose(Image.FLIP_LEFT_RIGHT)
				stack_array.append(np.asarray(cropped_u, dtype=np.float32))
				stack_array.append(np.asarray(cropped_v, dtype=np.float32))
			else: 
				img_array.append([cropped_u, cropped_v])
		if is_training:
			video_array.append(stack_array)
		else:
			for f_id in range(10):
				for [u, v] in img_array:
					if f_id != 0:
						u = multiView[f_id -1](u)
						v = multiView[f_id -1](v)
					stack_array.append(np.asarray(u, dtype=np.float32))
					stack_array.append(np.asarray(v, dtype=np.float32))
				video_array.append(stack_array)
				stack_array = []
	
	# frames = sorted(os.listdir(video_path))
	# selectedFrame = np.random.choice(frames)
	# print(selectedFrame)
	# frameId = frames.index(selectedFrame)
	# selectedFrames = [f for f in frames[frameId:frameId+stack]]

	# img = Image.open(os.path.join(video_path,selectedFrame))
	# width = img.size[0]
	# height = img.size[1]
	# startWidth = (width - 224)*np.random.random_sample()
	# startHeight = (height - 224)*np.random.random_sample()
	# chance = np.random.random()
	# img_array = []
	# for f in selectedFrames:
		# path_v = os.path.join(video_path, f)
		# path_u = path_v.replace('/v/', '/u/')
	
		# img_v = Image.open(path_v)
		# img_u = Image.open(path_u)
		
		# cropped_v = img_v.crop((startWidth, startHeight, startWidth+224, startHeight+224))
		# cropped_u = img_u.crop((startWidth, startHeight, startWidth+224, startHeight+224))
		# if chance > 0.5:
			# cropped_v = cropped_v.transpose(Image.FLIP_LEFT_RIGHT)
			# cropped_u = cropped_u.transpose(Image.FLIP_LEFT_RIGHT)
		# img_array.append(np.asarray(cropped_u, dtype=np.float32))
		# img_array.append(np.asarray(cropped_v, dtype=np.float32))
	print(np.array(video_array).shape)
	
	# print (img_array[0])
	
	# newFlow = img_array[0]*(40.0/255.0) - 40.0/2
	# print (newFlow)
	# print(np.mean(newFlow))
	# print(newFlow-np.mean(newFlow))
	
	
	
	
	
	