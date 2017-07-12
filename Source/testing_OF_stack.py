import numpy as np
import os
from PIL import Image
from cntk import load_model
from cntk.ops import softmax


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
	img1 = lambda x: x
	img1.__name__ = '-'
	img2 = lambda x: x.transpose(Image.ROTATE_90)
	img2.__name__ = '90'
	img3 = lambda x: x.transpose(Image.ROTATE_180)
	img3.__name__ = '180'
	img4 = lambda x: x.transpose(Image.ROTATE_270)
	img4.__name__ = '270'
	img5 = lambda x: x.transpose(Image.TRANSPOSE)
	img5.__name__ = 'trans'
	img6 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)
	img6.__name__ = 'flip-'
	img7 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
	img7.__name__ = 'flip90'
	img8 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
	img8.__name__ = 'flip180'
	img9 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
	img9.__name__ = 'flip270'
	img10 = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.TRANSPOSE)
	img10.__name__ = 'flipTrans'
	return [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10]
	
def getReescaleFlow():
	return lambda x: [y*(40.0/255.0) - 40.0/2 for y in x]
	
def transformFlow(u, v):
		# Reescale flow values
		u = reescaleFlow(u)
		v = reescaleFlow(v)
		# Reduce displacement field mean value
		meanFlow = np.mean([u, v])
		return (u - meanFlow), (v - meanFlow)
		
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
	
if __name__ == '__main__':

	train_map_file = os.path.join(data_dir, "ucfTrainTestlist", "trainlist01.txt")
	frames_dir	   = os.path.join(data_dir, "UCF-101_opticalFlow")
	stack = 10
	video_path = 'F:/TCC/Datasets/UCF-101_opticalFlow/v/v_TennisSwing_g01_c01'
	correctLabel = 91
	multiView = multiView()
	reescaleFlow = getReescaleFlow()
	
	frames = sorted(os.listdir(video_path))
	selectedFrames = []
	is_training = False
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
	width  = img.size[0]
	height = img.size[1]
	video_array = []
	# 25 for testing
	for frameStack in videoStack:
		startWidth  = (width - 224)*np.random.random_sample()
		startHeight = (height - 224)*np.random.random_sample()
		seq_array = []
		# 10 frames in the stack
		for path_v in frameStack:
			path_u = path_v.replace('/v/', '/u/')
			img_u = Image.open(path_u)
			img_v = Image.open(path_v)
			cropped_u = img_u.crop((startWidth, startHeight, startWidth+224, startHeight+224))
			cropped_v = img_v.crop((startWidth, startHeight, startWidth+224, startHeight+224))
			seq_array.append([cropped_u, cropped_v, path_v])
		# Making 10 stacks and adding them for testing
		for f_id in range(10):
			stack_array = []
			for [u, v, path] in seq_array:
				if f_id != 0:
					u = multiView[f_id](u)
					v = multiView[f_id](v)
				new_u, new_v = transformFlow(np.asarray(u, dtype=np.float32), 
												np.asarray(v, dtype=np.float32))
				stack_array.append(new_u)
				stack_array.append(new_v)
				# print(path, multiView[f_id].__name__)
			video_array.append(stack_array)
			# print(np.array(video_array).shape)
			
			
	video_array = np.array(video_array)
	print(video_array.shape)
	
	test_model = os.path.join("F:\TCC\Outputs\Output-ResNet34_videoOF_40%trained\Models", "ResNet_34_800.model")
	loaded_model = load_model(test_model)
	
	predictedLabels = dict()
	labelsConfidence = dict()
	for i, video in enumerate(video_array):
		output = loaded_model.eval({loaded_model.arguments[0]:video})
		predictions = softmax(np.squeeze(output)).eval()
		top_class = np.argmax(predictions)
		if top_class in predictedLabels.keys():
			predictedLabels[top_class] += 1
			labelsConfidence[top_class] += predictions[top_class] * 100
		else:
			predictedLabels[top_class] = 1
			labelsConfidence[top_class] = predictions[top_class] * 100
		if i%10==0:
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			print('\n{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabel, label, confidence))
			print('-------------')
		# print('{:^15} | {:^15} | {:^15.2f}%'.format(correctLabel, top_class, predictions[top_class]*100))
		
	label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
	print('\n{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabel, label, confidence))
		
	
	
	