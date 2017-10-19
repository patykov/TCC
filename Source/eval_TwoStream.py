import argparse
from cntk import load_model, UnitType, Communicator
from cntk.device import gpu, try_set_default_device, DeviceDescriptor
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.ops import input_variable, softmax, splice, combine, reduce_mean
import numpy as np
import os
from sklearn.svm import LinearSVC

# Create a minibatch source.
def create_of_mb_source(map_files, num_channels, image_height, image_width, num_classes):
	transforms = [#xforms.crop(crop_type='MultiView10', crop_size=224),
				xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
	
	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): StreamDef(field='image', transforms=transforms),
				   "label"+str(i): StreamDef(field='label', shape=num_classes)}
		sources.append(ImageDeserializer(map_file, StreamDefs(**streams)))
	
	return MinibatchSource(sources, max_sweeps=1, randomize=False)

def create_rgb_mb_source(map_file, num_channels, image_height, image_width, num_classes):
	transforms = [
		xforms.crop(crop_type='center', crop_size=224),
		xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	source = [
		ImageDeserializer(map_file, StreamDefs(
				features = StreamDef(field='image', transforms=transforms),
				labels   = StreamDef(field='label', shape=num_classes)))
	]

	return MinibatchSource(source, max_sweeps=1, randomize=False)
	
def getLastmodel(output_dir, model_name):
	trained_models = [f for f in os.listdir(output_dir) if f.endswith('.dnn')]
	if len(trained_models) > 0:
		trained_models = sorted(trained_models, key=lambda x: int(x.split(model_name+'_')[1].split('.dnn')[0]))
		print(trained_models[-1])
		return os.path.join(output_dir, trained_models[-1])
	return None
	

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
def eval_TwoStream(trained_RGB_model, trained_OF_model, OF_test_mapFiles, RGB_test_mapFile, output_file):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_classes	 = 101
	
	# Assure map files correct order
	map_files = sorted(OF_test_mapFiles, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	# Create test readers
	RGB_reader = create_rgb_mb_source(RGB_test_mapFile, 3, image_height, image_width, num_classes)
	OF_reader = create_of_mb_source(map_files, 1, image_height, image_width, num_classes)
	
	features_RGB = RGB_reader['features']
	OF_input_map = {}
	for i in range(20):
		OF_input_map[trained_OF_model.find_by_name("input_"+str(i))] = OF_reader.streams["feature"+str(i)]

	with open(OF_test_mapFiles[0], 'r') as file:
		lines = file.readlines()
	max_samples = len(lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	sample_count = 0.0
	clf = LinearSVC(random_state=0)
	results = ''
	with open(output_file, 'a') as file:
		while sample_count < max_samples:
			id_correctLabel = int(sample_count/25)
			sample_count += 25
			
			### OF eval ###
			OF_mb = OF_reader.next_minibatch(25, input_map=OF_input_map)
			OF_predictedLabels = dict((key, 0) for key in range(num_classes))
			OF_labelsConfidence = dict((key, 0) for key in range(num_classes))
			OF_output = trained_OF_model.eval(OF_mb)
			OF_predictions = softmax(np.squeeze(OF_output)).eval()
			
			### RGB eval ###
			RGB_mb = RGB_reader.next_minibatch(25)
			RGB_predictedLabels = dict((key, 0) for key in range(num_classes))
			RGB_labelsConfidence = dict((key, 0) for key in range(num_classes))
			RGB_output = trained_RGB_model.eval({trained_RGB_model.arguments[0]:RGB_mb[features_RGB]})
			RGB_predictions = softmax(np.squeeze(RGB_output)).eval()
						
			predictions = OF_predictions+RGB_predictions
			predictedLabels = dict((key, 0) for key in range(num_classes))
			labelsConfidence = dict((key, 0) for key in range(num_classes))
			for i, c in enumerate([np.argmax(p) for p in predictions]):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			
			# print('{:^15} | {:^15} | {:^15} | {:^15}\n'.format(correctLabels[id_correctLabel], RGB_label, OF_label, two_label))

			results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabels[id_correctLabel], label, confidence)
			if sample_count%500 == 0:
				print('{:.2f}% samples evaluated!'.format((sample_count/max_samples)*100))
				file.write(results)
				results = ''
		file.write(results)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default='/hdfs/pnrsy/t-pakova')
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
	args = parser.parse_args()
	
	if args.device is not None:
		try_set_default_device(gpu(args.device))
	
	# Paths
	data_dir = args.datadir
	model_name   = "VGG16_twoStream_sumSoftmax_"
	if args.logdir is not None:
		logFile = args.logdir
	map_dir = os.path.join(data_dir, "OF_mapFiles-forLaterXforms")
	output_dir = os.path.join(data_dir, model_name)
	OF_test_mapFiles  = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f])
	RGB_test_mapFiles = 'F:/TCC/Datasets/RGB_mapFiles_forLaterXforms/testMap_1.txt'
	output_file	   = os.path.join(data_dir, "eval_{}.txt".format(model_name))
	# output_file	   = os.path.join(output_dir, "eval_{}.txt".format(model_name))
	
	# Load Spatial VGG
	trained_RGB_model = load_model('F:/TCC/Models/VGG_videoRGB-part2')
	# trained_RGB_model = combine([trained_model.outputs[0].owner])
	# Load Temporal VGG
	trained_OF_model = load_model('F:/TCC/Models/philly/VGG16_videoOF_two_20.dnn')
	trained_OF_model = combine([trained_OF_model.outputs[0].owner])
	
	if not (os.path.exists(output_file)):
		# raise Exception('The file {} already exist.'.format(output_file))
		with open(output_file, 'w') as results_file:
			results_file.write('{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	# evaluate model and write out the desired output
	eval_TwoStream(trained_RGB_model, trained_OF_model, OF_test_mapFiles, RGB_test_mapFiles, output_file)
	
	# print("Done. Wrote output to %s" % output_file)
	
	Communicator.finalize()