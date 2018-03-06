import argparse
import cntk as C
import numpy as np
import os


# Create a minibatch source.
def create_of_mb_source(map_files, num_channels, image_height, image_width, num_classes):
	transforms = [
		C.io.transforms.crop(crop_type='center', crop_size=224),
		C.io.transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	print(map_files)
	
	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): C.io.StreamDef(field='image', transforms=transforms),
				    "label"+str(i): C.io.StreamDef(field='label', shape=num_classes)}
		sources.append(C.io.ImageDeserializer(map_file, C.io.StreamDefs(**streams)))
	
	return C.io.MinibatchSource(sources, max_sweeps=1, randomize=False)

def create_rgb_mb_source(map_file, num_channels, image_height, image_width, num_classes):
	transforms = [
		C.io.transforms.crop(crop_type='center', crop_size=224),
		C.io.transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	streams = {"features": C.io.StreamDef(field='image', transforms=transforms),
			   "labels": C.io.StreamDef(field='label', shape=num_classes)}
	source = [C.io.ImageDeserializer(map_file, C.io.StreamDefs(**streams))]
	
	return C.io.MinibatchSource(source,max_sweeps=1, randomize=False)

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
def eval_TwoStream(trained_RGB_model, trained_OF_model, OF_test_mapFiles, RGB_test_mapFile, num_inputs, output_file):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_classes	 = 101
	
	# Create test readers
	RGB_reader = create_rgb_mb_source(RGB_test_mapFile, 3, image_height, image_width, num_classes)
	OF_reader = create_of_mb_source(OF_test_mapFiles, 3, image_height, image_width, num_classes)
	
	# Create input maps
	RGB_input_map = {trained_RGB_model.find_by_name("data"): RGB_reader.streams["features"]}
	OF_input_map = {}
	for i in range(num_inputs):
		OF_input_map[trained_OF_model.find_by_name("input_"+str(i))] = OF_reader.streams["feature"+str(i)]

	with open(OF_test_mapFiles[0], 'r') as file:
		of_lines = file.readlines()
	with open(RGB_test_mapFile, 'r') as file:
		rgb_lines = file.readlines()
	assert(len(of_lines) == len(rgb_lines))
	max_samples = len(of_lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = of_lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	sample_count = 0.0
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
			#OF_predictions = C.ops.softmax(np.squeeze(OF_output)).eval()
			
			### RGB eval ###
			RGB_mb = RGB_reader.next_minibatch(25, input_map=RGB_input_map)
			RGB_predictedLabels = dict((key, 0) for key in range(num_classes))
			RGB_labelsConfidence = dict((key, 0) for key in range(num_classes))
			RGB_output = trained_RGB_model.eval(RGB_mb)
			#RGB_predictions = C.ops.softmax(np.squeeze(RGB_output)).eval()
						
			#predictions = OF_predictions+RGB_predictions
			predictions = C.ops.softmax((np.squeeze(OF_output)+np.squeeze(RGB_output))/2).eval()
			
			predictedLabels = dict((key, 0) for key in range(num_classes))
			labelsConfidence = dict((key, 0) for key in range(num_classes))
			for i, c in enumerate([np.argmax(p) for p in predictions]):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabels[id_correctLabel], label, confidence)
			if sample_count%500 == 0:
				print('{:.2f}% samples evaluated!'.format((sample_count/max_samples)*100))
				file.write(results)
				results = ''
		file.write(results)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-file_name', help='Name of output file.', 
						required=True)
	parser.add_argument('-rgb_model_path', help='Path to the espacial model', 
						required=True)
	parser.add_argument('-of_model_path', help='Path to the of temporal model', 
						required=False, default=None)
	parser.add_argument('-rgbdiff_model_path', help='Path to the rgbdiff temporal model', 
						required=False, default=None)
	parser.add_argument('-output_dir', help='Output directory',
						required=False, default="E:/TCC/Results/new")
	parser.add_argument('-data_dir', help='Data directory.', 
						required=False, default="E:/TCC/Datasets")
	parser.add_argument('-rgb_data_dir', help='Directory relative to data_dir with rgb data.', 
						required=False, default="RGB_mapFiles")
	parser.add_argument('-of_data_dir', help='Directory relative to data_dir with of data.', 
						required=False, default="OF_mapFiles_half")
	parser.add_argument('-rgbdiff_data_dir', help='Directory relative to data_dir with rgbdif data.', 
						required=False, default="RGBdiff_mapFiles")
	args = parser.parse_args()
	
	if (args.of_model_path and args.rgbdiff_model_path) is not None:
		raise valueError("Please select only one temporal network")
	elif args.of_model_path is not None:
		num_inputs = 20
		OF_map_dir = os.path.join(args.data_dir, args.of_data_dir)
		of_model_path = args.of_model_path
	else:
		num_inputs = 5
		OF_map_dir = os.path.join(args.data_dir, args.rgbdiff_data_dir)
		of_model_path = args.rgbdiff_model_path
		
	RGB_map_dir = os.path.join(args.data_dir, args.rgb_data_dir)
	output_file_path = os.path.join(args.output_dir, "eval_{}.txt".format(args.file_name))
	OF_test_mapFiles = sorted([os.path.join(OF_map_dir, f) for f in os.listdir(OF_map_dir) if 'test' in f])
	RGB_test_mapFile = sorted([os.path.join(RGB_map_dir, f) for f in os.listdir(RGB_map_dir) if 'test' in f])[0]

	
	# Load Spatial VGG
	trained_RGB_model = C.load_model(args.rgb_model_path)
	trained_RGB_model = C.ops.combine([trained_RGB_model.outputs[0].owner])
	
	# Load Temporal VGG
	trained_OF_model = C.load_model(of_model_path)
	trained_OF_model = C.ops.combine([trained_OF_model.outputs[0].owner])
	
	if os.path.exists(output_file_path):
		raise Exception('The file {} already exist.'.format(output_file_path))
	with open(output_file_path, 'w') as results_file:
		results_file.write('{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	# evaluate model and write out the desired output
	eval_TwoStream(trained_RGB_model, trained_OF_model, OF_test_mapFiles, RGB_test_mapFile, num_inputs, output_file_path)
	
	print("Done. Wrote output to %s" % output_file_path)
