import cntk as C
from cntk.layers import Activation, AveragePooling, BatchNormalization, Convolution, Convolution2D, Dense, Dropout, default_options, For, MaxPooling, Sequential


def from_scratch(model_name, input_var, num_classes, dropout=0.5):
	if model_name == "resnet34":
		z = create_resnet34(input_var, num_classes)
	if model_name == "vgg16":
		z = create_vgg16(input_var, num_classes, dropout)
	else:
		raise RuntimeError("Unknown model name!")
	
	return z

def clone_layers(base_model_path, feature_node_name, last_node_name, freeze):
	base_model   = C.load_model(base_model_path)
	feature_node = C.logging.graph.find_by_name(base_model, feature_node_name)
	last_node	 = C.logging.graph.find_by_name(base_model, last_node_name)

	# Clone the desired layers
	cloned_layers = C.combine([last_node.owner]).clone(C.CloneMethod.freeze 
		if freeze else C.CloneMethod.clone, {feature_node: C.placeholder(name='features')})
	
	return cloned_layers
			
def get_multiple_inputs(num_inputs, num_channels, image_height, image_width):
	inputs = []
	for c in range(num_inputs):
		inputs.append(C.input_variable((num_channels, image_height, image_width), name='input_{}'.format(c)))
	return inputs
	
def get_OF_pre_input(inputs):
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	new_inputs = [(i-C.ops.reduce_mean(i, axis=[1,2])) for i in input_reescaleFlow]
	pre_input = C.ops.splice(*(i for i in new_inputs), axis=0, name='pre_input')
	return pre_input
	
def get_diff_pre_input(inputs):
	new_inputs = [inputs[i] - inputs[i+1] for i in range(len(inputs)-1)]
	pre_input = C.ops.splice(*(i for i in new_inputs), axis=0, name='pre_input')
	return pre_input

def networkDict(z, label_var, inputs):
	# Loss and metric
	ce = C.losses.cross_entropy_with_softmax(z, label_var)
	pe = C.metrics.classification_error(z, label_var)
	
	features = {}
	for i, input in enumerate(inputs):
		features['feature'+str(i)] = input
	
	return dict({
		'model': z,
		'ce' : ce,
		'pe' : pe,
		'label': label_var}, 
		**features)
	
def print_layers(z):
	node_outputs = C.logging.get_node_outputs(z)
	for l in node_outputs: print("  {0} {1}".format(l.name, l.shape))

def create_vgg16(feature_var, num_classes, dropout_value=0.5):

    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU)
            For(range(2), lambda i: [
                Convolution2D((3,3), 64, name='conv1_{}'.format(i)),
                Activation(activation=C.ops.relu, name='relu1_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool1'),

            For(range(2), lambda i: [
                Convolution2D((3,3), 128, name='conv2_{}'.format(i)),
                Activation(activation=C.ops.relu, name='relu2_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool2'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 256, name='conv3_{}'.format(i)),
                Activation(activation=C.ops.relu, name='relu3_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool3'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv4_{}'.format(i)),
                Activation(activation=C.ops.relu, name='relu4_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool4'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv5_{}'.format(i)),
                Activation(activation=C.ops.relu, name='relu5_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool5'),

            Dense(4096, name='fc6'),
            Activation(activation=C.ops.relu, name='relu6'),
            Dropout(dropout_value, name='drop6'),
            Dense(4096, name='fc7'),
            Activation(activation=C.ops.relu, name='relu7'),
            Dropout(dropout_value, name='drop7'),
            Dense(num_classes, name='fc8')
            ])(feature_var)

    return z

def create_resnet34(input, num_classes):
	c_map = [64, 128, 256, 512]
	num_layers = [3, 3, 5, 2]

	conv = conv_bn_relu(input, (7,7), c_map[0], (2,2))
	maxPool = MaxPooling((3,3), (2,2), pad=True)(conv)
	r1 = resnet_basic_stack(maxPool, num_layers[0], c_map[0])

	r2_1 = resnet_basic_inc(r1, c_map[1])
	r2_2 = resnet_basic_stack(r2_1, num_layers[1], c_map[1])

	r3_1 = resnet_basic_inc(r2_2, c_map[2])
	r3_2 = resnet_basic_stack(r3_1, num_layers[2], c_map[2])
	
	r4_1 = resnet_basic_inc(r3_2, c_map[3])
	r4_2 = resnet_basic_stack(r4_1, num_layers[3], c_map[3])

	# Global average pooling and output
	pool = AveragePooling(filter_shape=(7,7))(r4_2) 
	z = Dense(num_classes)(pool)
	return z

#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=C.initializer.he_normal()):
	c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
	r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
	return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=C.initializer.he_normal()):
	r = conv_bn(input, filter_size, num_filters, strides, init) 
	return C.ops.relu(r)

def resnet_basic(input, num_filters):
	c1 = conv_bn_relu(input, (3,3), num_filters)
	c2 = conv_bn(c1, (3,3), num_filters)
	p  = c2 + input
	return C.ops.relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
	c1 = conv_bn_relu(input, (3,3), num_filters, strides)
	c2 = conv_bn(c1, (3,3), num_filters)
	s  = conv_bn(input, (1,1), num_filters, strides)
	p  = c2 + s
	return C.ops.relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
	assert (num_stack_layers >= 0)
	l = input 
	for _ in range(num_stack_layers): 
		l = resnet_basic(l, num_filters)
	return l
	
	