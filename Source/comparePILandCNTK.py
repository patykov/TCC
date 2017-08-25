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

    def __init__(self, map_file, dataDir, image_width, image_height, num_channels,
                    label_count, is_training=True, classMapFile=None):
        '''
        Load video file paths and their corresponding labels.
        '''
        self.map_file        = map_file
        self.label_count     = label_count
        self.width           = image_width
        self.height          = image_height
        self.sequence_length = 250
        self.channel_count   = num_channels
        self.is_training     = is_training
        self.video_files     = []
        self.targets         = []
        self.myAuxList       = [None]*self.label_count

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
        workList.sort(key=len, reverse=True)
        aux = list(itertools.izip_longest(*workList))
        self.indices = [x for x in itertools.chain(*list(itertools.izip_longest(*workList))) if x != None]
    
    def formatImg(self, img):
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

        inputs  = np.empty(shape=(current_batch_size, self.sequence_length, self.channel_count, self.height, self.width), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.sequence_length, self.label_count), dtype=np.float32)
        for idx in xrange(self.batch_start, batch_end):
            index = self.indices[idx]
            inputs[idx - self.batch_start, :, :, :, :] = self._select_features(self.video_files[index])
            targets[idx - self.batch_start, :, :]      = self.targets[index]
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
        w, h = self.getSizes(img, 256)                     # Get new size so the smallest side equals 256
        t1  = img.resize((w, h), Image.ANTIALIAS)          # Upscale so the min size equals 256
        t2  = self.randomCrop(t1, 256, 256)                # Crop random 256 square
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
        
        # MultiView10
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = img.crop((img.size[0]/2 - self.width/2, img.size[1]/2 - self.height/2, 
                        img.size[0]/2 + self.width/2, img.size[1]/2 + self.height/2)) # center
        img2 = img.crop((0, 0,  self.width, self.height)) # top left
        img3 = img.crop((0, img.size[1] - self.height, self.width, img.size[1])) # bottom left
        img4 = img.crop((img.size[0] - self.width, 0, 
                        img.size[0], self.height)) #top right
        img5 = img.crop((img.size[0] - self.width, img.size[1] - self.height, 
                        img.size[0], img.size[1])) # bottom right
        img6 = flip_img.crop((flip_img.size[0]/2 - self.width/2, flip_img.size[1]/2 - self.height/2, 
                            flip_img.size[0]/2 + self.width/2, flip_img.size[1]/2 + self.height/2)) # flip center
        img7 = flip_img.crop((0, 0, self.width, self.height)) # flip top left
        img8 = flip_img.crop((0, flip_img.size[1] - self.height, 
                            self.width, flip_img.size[1])) # flip bottom left
        img9 = flip_img.crop((flip_img.size[0] - self.width, 
                            0, flip_img.size[0], self.height)) # flip top right
        img10 = flip_img.crop((flip_img.size[0] - self.width, flip_img.size[1] - self.height, 
                            flip_img.size[0], flip_img.size[1])) # flip bottom right
                            
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
	
	
if __name__ == '__main__':
	set_default_device(gpu(0))

	# define location of model and data and check existence
	base_folder = 'E:\TCC'
	model = 'ResNet_34_ucf101_rgb_py2'
	model_dir    = os.path.join(base_folder, "Models")
	dataDir      = os.path.join(base_folder, "DataSets", "UCF-101_rgb")
	map_file_mb  = os.path.join(base_folder, "DataSets", "TestMapFiles01_RGB.txt", "test_map01_RGB0_0.txt")
	
	# create minibatch source
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_output_classes = 101

	with open(map_file, 'r') as file:
		lines = file.readlines()
	test_reader = VideoReader(map_file, dataDir, image_width, image_height, num_channels,
							label_count, is_training=False, classMapFile=None)
		
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

				