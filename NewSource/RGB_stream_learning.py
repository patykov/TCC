import argparse
from cntk.train.training_session import *
import stream_learning as lr
import model_configuration as mc
import numpy as np
import os
import readers as rd
import shutil


# Create a minibatch source.
def create_video_mb_source(map_file, num_channels, image_height, image_width, num_classes, max_epochs, 
                            is_training=True):
	if is_training:
		transforms = [
			xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2),
			xforms.crop(crop_type='randomside', crop_size=224)
			]
		randomize = True
	else:
		transforms = [xforms.crop(crop_type='MultiView10', crop_size=224)]
		randomize = False
	
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
				features = StreamDef(field='image', transforms=transforms),
				labels   = StreamDef(field='label', shape=num_classes))), 
				max_sweeps=max_epochs, randomize=randomize)

	
# Trains a RGB stream 
def train_model(baseNetworkPath, train_mapFile, output_dir, log_file, model_name, num_epochs,
				mb_size, profiling=False):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_classes	 = 101
	
	# Base model layers for ImageNet fine-tuning
	feature_node_name = 'data'
	last_node_name = 'drop7'
	
	# Create network:
	network = mc.get_RGB_fine_tuning(baseNetworkPath, feature_node_name, last_node_name,
						num_classes, num_channels, image_height, image_width)
	
	# Get learner
	lr_per_mb = lr.get_default_rgb_lr()
	learner = lr.define_learner(model, lr_per_mb, distributed=True)
	
	# Get reader
	transforms = rd.get_default_transforms(num_channels, image_height, image_width is_training=True)
	train_reader = rd.create_mb_source(train_mapFile, transforms, num_classes, max_epochs, randomize=True)
	
	# Train 
	new_model = 



if __name__ == '__main__':
    """ Parses input and calls train."""
    # Adding arguments
    parser = argparse.ArgumentParser()
    # Directory to where input data (old model + new data) is located
    parser.add_argument('-datadir', '--datadir', help='Directory for input data', required=False)
    parser.add_argument('-outputdir', '--outputdir', help='Directory where results will be saved',
                        required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true',
                        default=False)
    parser.add_argument('-baseModelPath', 'baseModelPath', 'Path to the base model.', default=None)
    parser.add_argument('-mapdir', 'mapdir', 'Directory where the mapping files are.', default=None)
    parser.add_argument('-num_epochs', '--num_epochs', help='Total number of epochs to train', type=int,
                        required=False, default='3')
    parser.add_argument('-mb_size', '--mb_size', help='Minibatch size', type=int, required=False,
                        default='256')
    parser.add_argument('newNetworkName', '--newNetworkName', "Name of the new network.", default="RGB")
	parser.add_argument('model_name', '--model_name', "Name of the model type.", default="VGG16")
	args = parser.parse_args()
	
	# Paths
	if args.baseModelPath is None:
		args.baseModelPath = os.path.join(args.datadir, 'VGG16_ImageNet_CNTK.model')
	if args.mapdir is None:
		args.mapdir = os.path.join(args.datadir, "RGB_mapFiles")
	
	newModelName = "{}_{}{}".format(args.model_name, args.newNetworkName, '')
	output_dir = os.path.join(args.outputdir, newModelName)
	train_mapFile = [f for f in os.listdir(args.mapdir) if 'train' in f]
	new_model_file = os.path.join(output_dir, newModelName+'.dnn')

	# if os.path.exists(output_dir):
		# shutil.rmtree(output_dir)
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
		
	trained_model = train_model(baseNetworkPath = args.baseNetworkPath, 
								train_mapFile = train_mapFile, 
								output_dir = output_dir, 
								log_file = args.logdir, 
								model_name = newModelName, 
								profiling = args.profile,
								num_epochs = args.num_epochs
								mb_size = args.mb_size)
	trained_model.save(new_model_file)

	print("Stored trained model at %s" % new_model_file)
	
	Communicator.finalize()
