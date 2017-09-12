import argparse
from cntk import data_parallel_distributed_learner, load_model, Trainer, UnitType, Axis
from cntk.debugging import start_profiler, stop_profiler
from cntk.device import gpu, try_set_default_device
from cntk.io import MinibatchSource, ImageDeserializer
from cntk.layers import Dense
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
import zipfile

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))

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
		self.sequence_length = 1
		self.channel_count	 = num_channels
		self.is_training	 = is_training
		self.video_files	 = []
		self.targets		 = []
		self.myAuxList		 = [None]*self.label_count
        self.dataDir       = zipfile.ZipFile(os.path.join(dataDir, 'UCF-101_rgb.zip'), 'r')

        if not self.is_training:
            self.sequence_length = 250
            self.getClassDict(classMapFile)
        
        with open(map_file, 'r') as file:
            for row in file:
                if self.is_training:
                    [video_path, label] = row.replace('\n','').split(' ')
                else:
                    video_path, label = self.getTestClass(row)
                self.video_files.append(video_path)
                target = [0.0] * self.label_count
                target[int(label)-1] = 1.0
                self.targets.append(target)
                if self.myAuxList[int(label)-1] == None:
                    self.myAuxList[int(label)-1] = [len(self.targets)-1]
                else:
                    self.myAuxList[int(label)-1].append(len(self.targets)-1)

        self.indices = np.arange(len(self.video_files))
        self.groupByTarget()
        self.reset()

    def getClassDict(self, classMapFile):
        # Getting class id for test files
        self.classMap = dict()
        with open(classMapFile, 'r') as file:
            for line in file:
                [label, className] = line.replace('\n', '').split(' ')
                self.classMap[className] = label

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
        grouped = list(itertools.zip_longest(*workList))
        np.random.shuffle(grouped)
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
		listFiles = [f for f in self.dataDir.namelist() if video_path in f]
        frames = sorted(listFiles)[1:] # Removing dir path
		selectedFrames = []

		if self.is_training:
			video_frames = [self._train_transform(np.random.choice(frames))]
		else:
			length = self.sequence_length/10
			ids = np.linspace(0, len(frames), num=length, dtype=np.int32, endpoint=False)
			selectedFrames = [frames[i] for i in ids]
			video_frames = list(itertools.chain(*[self._test_transform(f) for f in selectedFrames]))
	
		return video_frames

	def _train_transform(self, image_path):
		# load image
		img = Image.open(self.dataDir.open(image_path))
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
		img = Image.open(self.dataDir.open(image_path))
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

	cloned_out = cloned_layers(input_features)
	z		   = Dense(num_classes, activation=None, name='fc101') (cloned_out)
	return z
	
# Trains a transfer learning model
def train_model(network_path, train_reader, output_dir, log_file, profiling=True):
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
	z = create_model(base_model, 'data', 'drop7', num_classes, input_var)
	
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
	local_learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, unit_gain=False,
                                l2_regularization_weight = l2_reg_weight)
    learner = data_parallel_distributed_learner(local_learner, num_quantization_bits = 32)
	trainer = Trainer(z, (ce, pe), learner, progress_writers)
	
	with open(logFile, 'a') as file:
		file.write('\nVGG + Freeze + allTransforms + videoTestMinibatch')
		file.write('\nMinibatch_size = {}\n'.format(minibatch_size))
	
	sample_count = trainer.total_number_of_samples_seen
	last_epoch = sample_count/train_reader.size()
	print('Total number of samples seen: {} | Last epoch: {}\n'.format(sample_count, last_epoch))
	
	if profiling:
        start_profiler(dir=output_dir, sync_gpu=True)
	
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
	
	if profiling:
        stop_profiler()
		
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
    results = ''
    with open(output_file, 'a') as file:
        while sample_count < test_reader.size():
            videos, labels, current_minibatch = test_reader.next_minibatch(1)
            sample_count += current_minibatch
            predictedLabels = dict((key, 0) for key in xrange(num_classes))
            labelsConfidence = dict((key, 0) for key in xrange(num_classes))
            correctLabel = [j for j,v in enumerate(labels[0][0]) if v==1.0][0]
            output = loaded_model.eval({loaded_model.arguments[0]:videos[0]})
            predictions = softmax(np.squeeze(output)).eval()
            top_classes = [np.argmax(p) for p in predictions]
            for i, c in enumerate(top_classes):
                predictedLabels[c] += 1 #Melhorar
                labelsConfidence[c] += predictions[i][c]
            label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
            results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabel, label, confidence)
            if sample_count%50 == 0:
                file.write(results)
                results = ''
        file.write(results)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('-logdir', help="""Directory where all the logs should be stored.""", required=False, default=None)
    parser.add_argument('-datadir', help="""Directory where all the datasets are stored.""", required=False, default=None)
    parser.add_argument('-outputdir', help="""Directory where all the models should be stored.""", required=False, default=None)
    parser.add_argument('-profile', help="""Whether to activate profiler or not.""", required=False, default=True)
    parser.add_argument('-device', '--device', type=int, help="""Force to run the script on a specified device""", required=False, default=None)
    args = parser.parse_args()

    # Model dimensions
    image_height  = 224
    image_width   = 224
	num_channels  = 3
    num_classes   = 101
    
    #For training
    newModelName   = "VVG16_videoRGB-philly"
    dataDir = args.datadir
    if args.device is not None:
        try_set_default_device(gpu(args.device))
    if args.outputdir is not None:
        outputDir = args.outputdir
    if args.logdir is not None:
        logFile = args.logdir
    train_map_file = os.path.join(dataDir, "trainlist01.txt")
    new_model_file = os.path.join(outputDir, newModelName+'.model')
	network_path   = os.path.join(dataDir, "VGG16_ImageNet_CNTK.model")
	
    #For evaluation
    test_map_file  = os.path.join(dataDir, "testlist01.txt")
    class_map_file = os.path.join(dataDir, "classInd.txt")
    output_file    = os.path.join(outputDir, "eval_{}.txt".format(newModelName))
    
    ### Training ###
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    train_reader = VideoReader(train_map_file, dataDir, image_width, image_height, num_channels,
                                num_classes, is_training=True)
    trained_model = train_model(network_path, train_reader, outputDir, logFile, args.profile)
    trained_model.save(new_model_file)
    
    ## Evaluation ###
    with open(output_file, 'w') as results_file:
        results_file.write('{:<15} | {:<15} | {:<15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
    
    test_reader = VideoReader(test_map_file, dataDir, image_width, image_height, num_channels, 
                                num_classes, is_training=False, classMapFile=class_map_file)
    # evaluate model and write out the desired output
    eval_and_write(trained_model, test_reader, output_file)
    
    # Must call MPI finalize when process exit without exceptions
    Communicator.finalize()
	