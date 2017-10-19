import argparse
from cntk import load_model, UnitType, Communicator
from cntk.device import gpu, try_set_default_device, DeviceDescriptor
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.ops import input_variable, softmax, splice, combine, reduce_mean
import numpy as np
from models import *
import os

# Create a minibatch source.
def create_of_mb_source1(map_files, num_channels, image_height, image_width, num_classes, is_training=True):

	transforms = [xforms.crop(crop_type='MultiView10')]
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	transforms += [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]

	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): StreamDef(field='image', transforms=transforms),
				   "label"+str(i): StreamDef(field='label', shape=num_classes)}
		sources.append(ImageDeserializer(map_file, StreamDefs(**streams)))

	return MinibatchSource(sources, max_sweeps=1, randomize=False)
	
def create_of_mb_source2(map_files, num_channels, image_height, image_width, num_classes, is_training=True):

	transforms = [xforms.crop(crop_type='MultiView10')]
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	transforms += [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]

	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = [
		ImageDeserializer(map_files[0], StreamDefs(
				features1 = StreamDef(field='image', transforms=transforms),
				labels1	  = StreamDef(field='label', shape=num_classes))),	 
		ImageDeserializer(map_files[1], StreamDefs(
				features2 = StreamDef(field='image', transforms=transforms),
				labels2	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[2], StreamDefs(
				features3 = StreamDef(field='image', transforms=transforms),
				labels3	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[3], StreamDefs(
				features4 = StreamDef(field='image', transforms=transforms),
				labels4	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[4], StreamDefs(
				features5 = StreamDef(field='image', transforms=transforms),
				labels5	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[5], StreamDefs(
				features6 = StreamDef(field='image', transforms=transforms),
				labels6	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[6], StreamDefs(
				features7 = StreamDef(field='image', transforms=transforms),
				labels7	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[7], StreamDefs(
				features8 = StreamDef(field='image', transforms=transforms),
				labels8	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[8], StreamDefs(
				features9 = StreamDef(field='image', transforms=transforms),
				labels9	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[9], StreamDefs(
				features10 = StreamDef(field='image', transforms=transforms),
				labels10   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[10], StreamDefs(
				features11 = StreamDef(field='image', transforms=transforms),
				labels11   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[11], StreamDefs(
				features12 = StreamDef(field='image', transforms=transforms),
				labels12   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[12], StreamDefs(
				features13 = StreamDef(field='image', transforms=transforms),
				labels13   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[13], StreamDefs(
				features14 = StreamDef(field='image', transforms=transforms),
				labels14   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[14], StreamDefs(
				features15 = StreamDef(field='image', transforms=transforms),
				labels15   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[15], StreamDefs(
				features16 = StreamDef(field='image', transforms=transforms),
				labels16   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[16], StreamDefs(
				features17 = StreamDef(field='image', transforms=transforms),
				labels17   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[17], StreamDefs(
				features18 = StreamDef(field='image', transforms=transforms),
				labels18   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[18], StreamDefs(
				features19 = StreamDef(field='image', transforms=transforms),
				labels19   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[19], StreamDefs(
				features20 = StreamDef(field='image', transforms=transforms),
				labels20   = StreamDef(field='label', shape=num_classes)))
	]

	return MinibatchSource(sources, max_sweeps=1, randomize=False)

def create_OF_model(num_classes, num_channels, image_height, image_width):
	# Input variables
	input_var = input_variable((num_channels, image_height, image_width))
	label_var = input_variable(num_classes)

	# create base model
	z_base = create_vgg16_2(input_var, num_classes)
	# node_outputs = get_node_outputs(z_base)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

	# Create new input with OF needed transforms
	inputs = []
	for c in range(num_channels):
		inputs.append(input_variable((1, image_height, image_width), name='input_{}'.format(c)))
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	input_reduceMean = [(i-reduce_mean(i, axis=[1,2])) for i in input_reescaleFlow]
	new_input = splice(*(i for i in input_reduceMean), axis=0, name='pre_input')
	
	z = z_base(new_input)
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))
	
	features = {}
	for i in range(num_channels):
		features['feature'+str(i)] = inputs[i]
	
	return dict({
		'model': z,
		'label': label_var}, 
		**features), {
		'model': z,
		'label': label_var,
		'feature1': inputs[0],
		'feature2': inputs[1],
		'feature3': inputs[2],
		'feature4': inputs[3],
		'feature5': inputs[4],
		'feature6': inputs[5],
		'feature7': inputs[6],
		'feature8': inputs[7],
		'feature9': inputs[8],
		'feature10': inputs[9],
		'feature11': inputs[10],
		'feature12': inputs[11],
		'feature13': inputs[12],
		'feature14': inputs[13],
		'feature15': inputs[14],
		'feature16': inputs[15],
		'feature17': inputs[16],
		'feature18': inputs[17],
		'feature19': inputs[18],
		'feature20': inputs[19]
	}

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
def eval_TwoStream(OF_test_mapFiles):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_classes	 = 101
	
	# Assure map files correct order
	map_files = sorted(OF_test_mapFiles, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	# Create test readers
	reader1 = create_of_mb_source1(map_files, 1, image_height, image_width, num_classes)
	reader2 = create_of_mb_source2(map_files, 1, image_height, image_width, num_classes)
	
	network1, network2 = create_OF_model(num_classes, 20, image_height, image_width)

	input_map1 = {}
	for i in range(20):
		input_map1[network1['model'].find_by_name("input_"+str(i))] = reader1.streams["feature"+str(i)]

	input_map2 = {
		network2['model'].input_0: reader2.streams.features1,
		network2['model'].input_1: reader2.streams.features2,
		network2['model'].input_2: reader2.streams.features3,
		network2['model'].input_3: reader2.streams.features4,
		network2['model'].input_4: reader2.streams.features5,
		network2['model'].input_5: reader2.streams.features6,
		network2['model'].input_6: reader2.streams.features7,
		network2['model'].input_7: reader2.streams.features8,
		network2['model'].input_8: reader2.streams.features9,
		network2['model'].input_9: reader2.streams.features10,
		network2['model'].input_10: reader2.streams.features11,
		network2['model'].input_11: reader2.streams.features12,
		network2['model'].input_12: reader2.streams.features13,
		network2['model'].input_13: reader2.streams.features14,
		network2['model'].input_14: reader2.streams.features15,
		network2['model'].input_15: reader2.streams.features16,
		network2['model'].input_16: reader2.streams.features17,
		network2['model'].input_17: reader2.streams.features18,
		network2['model'].input_18: reader2.streams.features19,
		network2['model'].input_19: reader2.streams.features20
	}
		
	with open(OF_test_mapFiles[0], 'r') as file:
		lines = file.readlines()
	max_samples = len(lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	sample_count = 0.0
	results = ''
	while sample_count < max_samples:
		id_correctLabel = int(sample_count/25)
		sample_count += 25
		
		### OF eval 1 ###
		mb1 = reader1.next_minibatch(1, input_map=input_map1)
		predictedLabels1 = dict((key, 0) for key in range(num_classes))
		labelsConfidence1 = dict((key, 0) for key in range(num_classes))
		output1 = network1['model'].eval(mb1)
		predictions1 = softmax(np.squeeze(output1)).eval()
		
		### OF eval 1 ###
		mb2 = reader2.next_minibatch(1, input_map=input_map2)
		predictedLabels2 = dict((key, 0) for key in range(num_classes))
		labelsConfidence2 = dict((key, 0) for key in range(num_classes))
		output2 = network2['model'].eval(mb2)
		predictions2 = softmax(np.squeeze(output2)).eval()
		
		import pdb
		pdb.set_trace()
		# predictions = OF_predictions+RGB_predictions
		# predictedLabels = dict((key, 0) for key in range(num_classes))
		# labelsConfidence = dict((key, 0) for key in range(num_classes))
		# for i, c in enumerate([np.argmax(p) for p in predictions]):
			# predictedLabels[c] += 1 #Melhorar
			# labelsConfidence[c] += predictions[i][c]
		# label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
		
		# print('{:^15} | {:^15} | {:^15} | {:^15}\n'.format(correctLabels[id_correctLabel], RGB_label, OF_label, two_label))

	


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

	
	
	# evaluate model and write out the desired output
	eval_TwoStream(OF_test_mapFiles)
	
	# print("Done. Wrote output to %s" % output_file)
	
	Communicator.finalize()