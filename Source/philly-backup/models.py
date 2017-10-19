from cntk.initializer import he_normal
from cntk.layers import Activation, AveragePooling, BatchNormalization, Convolution, Convolution2D, Dense, Dropout, default_options, For, MaxPooling, Sequential
from cntk.ops import element_times, relu


def create_vgg16(feature_var, num_classes):

    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU)
            For(range(2), lambda i: [
                Convolution2D((3,3), 64, name='conv1_{}'.format(i)),
                Activation(activation=relu, name='relu1_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool1'),

            For(range(2), lambda i: [
                Convolution2D((3,3), 128, name='conv2_{}'.format(i)),
                Activation(activation=relu, name='relu2_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool2'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 256, name='conv3_{}'.format(i)),
                Activation(activation=relu, name='relu3_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool3'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv4_{}'.format(i)),
                Activation(activation=relu, name='relu4_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool4'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv5_{}'.format(i)),
                Activation(activation=relu, name='relu5_{}'.format(i)),
            ]),
            MaxPooling((2,2), (2,2), name='pool5'),

            Dense(4096, name='fc6'),
            Activation(activation=relu, name='relu6'),
            Dropout(0.5, name='drop6'),
            Dense(4096, name='fc7'),
            Activation(activation=relu, name='relu7'),
            Dropout(0.5, name='drop7'),
            Dense(num_classes, name='fc8')
            ])(feature_var)

    return z


#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
	c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
	r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
	return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
	r = conv_bn(input, filter_size, num_filters, strides, init) 
	return relu(r)

def resnet_basic(input, num_filters):
	c1 = conv_bn_relu(input, (3,3), num_filters)
	c2 = conv_bn(c1, (3,3), num_filters)
	p  = c2 + input
	return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
	c1 = conv_bn_relu(input, (3,3), num_filters, strides)
	c2 = conv_bn(c1, (3,3), num_filters)
	s  = conv_bn(input, (1,1), num_filters, strides)
	p  = c2 + s
	return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
	assert (num_stack_layers >= 0)
	l = input 
	for _ in range(num_stack_layers): 
		l = resnet_basic(l, num_filters)
	return l 

#	
# Defines the residual network model for classifying images
#
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