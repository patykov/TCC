from __future__ import print_function
import numpy as np
import os
from PIL import Image
from cntk.device import try_set_default_device, gpu
from cntk import load_model, placeholder, Constant
from cntk import Trainer, UnitType
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk.layers import Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.ops import input, combine, softmax
from cntk.ops.functions import CloneMethod
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.logging import log_number_of_parameters, ProgressPrinter


################################################
################################################
# general settings
make_mode = False
freeze_weights = False
base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
tl_model_file = os.path.join(base_folder, "Output", "resnet18_take and pullBack")
output_file = os.path.join(base_folder, "Output", "predOutput.txt")
features_stream_name = 'features'
label_stream_name = 'labels'
new_output_node_name = "prediction"

# define base model location and characteristics
_base_model_file = os.path.join(base_folder, "Models", "ResNet_18.model")
_feature_node_name = "features"
_last_hidden_node_name = "z.x"
_image_height = 224
_image_width = 224
_num_channels = 3
_num_classes = 101
################################################
################################################


# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_width, image_height, num_channels, num_classes, randomize=True):
    transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')] 
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
            features =StreamDef(field='image', transforms=transforms),
            labels   =StreamDef(field='label', shape=num_classes))),
            randomize=randomize)


# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_features, freeze=False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(base_model_file)
    feature_node = find_by_name(base_model, feature_node_name)
    last_node    = find_by_name(base_model, last_hidden_node_name)

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Add new dense layer for class prediction
    feat_norm  = input_features - Constant(114)
    cloned_out = cloned_layers(feat_norm)
    z          = Dense(num_classes, activation=None, name=new_output_node_name) (cloned_out)

    return z


# Trains a transfer learning model
def train_model(base_model_file, feature_node_name, last_hidden_node_name,
                image_width, image_height, num_channels, num_classes, freeze=False):

    # Create the minibatch source and input variables
    minibatch_source = create_mb_source(train_map_file, image_width, image_height, num_channels, num_classes)
    image_input = input((num_channels, image_height, image_width))
    label_input = input(num_classes)

    # Define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source[features_stream_name],
        label_input: minibatch_source[label_stream_name]
    }

    # Instantiate the transfer learning model and loss function
    tl_model = create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, image_input, freeze)


    return tl_model



if __name__ == '__main__':
    try_set_default_device(gpu(0))

    # You can use the following to inspect the base model and determine the desired node names
    # node_outputs = get_node_outputs(load_model(_base_model_file))
    # for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

	trained_model = train_model(_base_model_file, _feature_node_name, _last_hidden_node_name,
								_image_width, _image_height, _num_channels, _num_classes, freeze=freeze_weights)
	trained_model.save(tl_model_file) 
    print("Done.")
	
	