import cntk as C
import model_configuration as mc


def get_OF_from_scratch(model_name, dropout=0.5, num_inputs=20, num_classes=101, 
						num_channels=1, image_height=224, image_width=224):
	# Input variables
	input_var = C.input_variable((num_channels*num_inputs, image_height, image_width))
	label_var = C.input_variable(num_classes)
	
	# Create base model
	z_base = mc.from_scratch(model_name, input_var, num_classes, dropout)

	# Create new input with OF needed transforms
	inputs = mc.get_multiple_inputs(num_inputs, num_channels, image_height, image_width)
	pre_input = mc.get_OF_pre_input(inputs)
	z = z_base(pre_input)
	
	return mc.networkDict(z, label_var, inputs)

def get_RGB_fine_tuning(base_model_path, feature_node_name='data', last_node_name='drop7',
						num_classes=101, num_channels=3, image_height=224, image_width=224):
	# Input variables
	input_var = C.input_variable((num_channels, image_height, image_width))
	label_var = C.input_variable(num_classes)
	
	# Clone layers from base model
	cloned_layers = mc.clone_layers(base_model_path, feature_node_name, last_node_name, freeze=True)

	# Add new dense layer for class prediction
	cloned_out = mc.cloned_layers(input_var)
	z = C.layers.Dense(num_classes, activation=None, name='fc{}'.format(num_classes)) (cloned_out)
	
	return mc.networkDict(z, label_var, [input_var])

def get_RGBdiff_fine_tuning(base_model_path, feature_node_name='conv1_1', last_node_name='fc101',
							num_inputs=5, num_classes=101, num_channels=3, image_height=224, 
							image_width=224):
	# Label variable
	label_var = C.input_variable(num_classes)
	
	# Input variable
	inputs = mc.get_multiple_inputs(num_inputs, num_channels, image_height, image_width)
	pre_input = mc.get_diff_pre_input(inputs)

	# Clone layers from base model
	cloned_layers = mc.clone_layers(base_model_path, feature_node_name, last_node_name, freeze=False)
	
	# Create new conv layer
	conv = C.layers.Convolution2D((3,3), 64, name='conv1_1', activation=None, pad=True, bias=True)
	new_conv = conv(pre_input)
	
	# New model
	z = mc.cloned_layers(new_conv)
	
	return mc.networkDict(z, label_var, inputs)
	
def create_model(stream_type, outputdir, baseModelPath, first_node_name, last_node_name, 
				num_channels, image_height=224, image_width=224, num_classes=101, num_inputs,
				dropout, newNetworkName, model_name):
		
	
	
	
if __name__ == '__main__':
    """ Parses input and creates an model."""
    # Adding model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-stream_type', '--stream_type', help='Type of model to be created', required=False)
    parser.add_argument('-outputdir', '--outputdir', help='Directory where the model will be saved',
                        required=False, default=None)
    parser.add_argument('-baseModelPath', 'baseModelPath', 'Path to the base model.', default=None)
    parser.add_argument('-first_node_name', 'first_node_name', 'Name of the first node name of the base model to be cloned', 
						required=False, default=None)
	parser.add_argument('-last_node_name', 'last_node_name', 'Name of the last node name of the base model to be cloned', 
						required=False, default=None)
    parser.add_argument('-num_channels', '--num_channels', help='Number of channels in the image input.', 
						type=int, required=False)
	parser.add_argument('-image_height', '--image_height', help='Height of the image input.', type=int, 
						required=False, default=224)
	parser.add_argument('-image_width', '--image_width', help='Width of the image input.', type=int, 
						required=False, default=224)
	parser.add_argument('-num_classes', '--num_classes', help='Number of classes in the model.', type=int, 
						required=False, default=101)
	parser.add_argument('-num_inputs', '--num_inputs', help='Number of inputs in the model.', type=int, 
						required=False)
	parser.add_argument('-dropout', '--dropout', help='Dropout value to be used in the model.', type=float, 
						required=False)
    parser.add_argument('newNetworkName', '--newNetworkName', "Name of the new network.", required=False, 
						default=None)
	parser.add_argument('model_name', '--model_name', "Name of the model type.", default="VGG16")
	args = parser.parse_args()
	
	create_model(**args)
	
	
