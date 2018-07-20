import argparse
import os
import numpy as np
import cntk as C

import basic_models as m
import model_configuration as mc


def get_OF_from_scratch(model_name='vgg16', dropout=0.9, num_inputs=20, num_classes=101,
                        num_channels=3, image_height=224, image_width=224):
    # Input variables
    input_var = C.input_variable((num_inputs, image_height, image_width))
    label_var = C.input_variable(num_classes)

    # Create base model
    z_base = m.from_scratch(model_name, input_var, num_classes, dropout)

    # Create new input with OF needed transforms
    inputs = mc.get_multiple_inputs(num_inputs, num_channels, image_height, image_width)
    flowRange = 40.0
    imageRange = 255.0
    gray_inputs = [i[0] for i in inputs]
    input_reescaleFlow = [i * (flowRange / imageRange) - flowRange / 2 for i in gray_inputs]
    new_inputs = [(i - C.ops.reduce_mean(i, axis=[1, 2])) for i in input_reescaleFlow]
    pre_input = C.ops.splice(*(i for i in new_inputs), axis=0, name='pre_input')
    z = z_base(pre_input)

    return mc.networkDict(z, label_var, inputs)


def get_RGB_fine_tuning(base_model_path=None, freeze=True, feature_node_name='data',
                        last_node_name='drop7', num_classes=101, num_channels=3, image_height=224,
                        image_width=224):
    # Input variables
    input_var = C.input_variable((num_channels, image_height, image_width), name=feature_node_name)
    label_var = C.input_variable(num_classes)

    # Clone layers from base model
    cloned_layers = mc.clone_layers(
        base_model_path,
        feature_node_name,
        last_node_name,
        freeze=freeze)

    # Add new dense layer for class prediction
    cloned_out = cloned_layers(input_var)
    z = C.layers.Dense(num_classes, activation=None, name='fc{}'.format(num_classes))(cloned_out)

    return mc.networkDict(z, label_var, [input_var])


def get_old_RGBdiff_fine_tuning(base_model_path=None, after_feature_node_name='conv1_1',
                                last_node_name='drop7', num_inputs=5, num_classes=101,
                                num_channels=3, image_height=224, image_width=224):

    # Label variable
    label_var = C.input_variable(num_classes)

    # Input variable
    inputs = mc.get_multiple_inputs(num_inputs, num_channels, image_height, image_width)
    new_inputs = [inputs[i] - inputs[i + 1] for i in range(len(inputs) - 1)]
    pre_input = C.ops.splice(*(i for i in new_inputs), axis=0, name='pre_input')

    # Clone layers from base model
    cloned_layers = mc.clone_layers(
        base_model_path,
        after_feature_node_name,
        last_node_name,
        freeze=False)

    # Create new first conv layer
    conv = C.layers.Convolution2D((3, 3), 64, name=after_feature_node_name,
                                  activation=None, pad=True, bias=True)
    new_conv = conv(pre_input)

    cloned_out = cloned_layers(new_conv)
    z = C.layers.Dense(num_classes, activation=None, name='fc{}'.format(num_classes))(cloned_out)

    return mc.networkDict(z, label_var, inputs)


def get_RGBdiff_fine_tuning(base_model_path=None, after_feature_node_name='conv1_1',
                            last_node_name='drop7', num_inputs=5, num_classes=101, num_channels=3,
                            image_height=224, image_width=224):

    # Label variable
    label_var = C.input_variable(num_classes)

    # Input variable
    inputs = mc.get_multiple_inputs(num_inputs, num_channels, image_height, image_width)
    new_inputs = [inputs[i] - inputs[i + 1] for i in range(len(inputs) - 1)]
    pre_input = C.ops.splice(*(i for i in new_inputs), axis=0, name='pre_input')

    # Clone layers from base model
    cloned_layers = mc.clone_layers(
        base_model_path,
        after_feature_node_name,
        last_node_name,
        freeze=False)

    # Create new first conv layer
    conv = C.layers.Convolution2D((3, 3), 64, name=after_feature_node_name,
                                  activation=None, pad=True, bias=True)
    new_conv = conv(pre_input)

    # Copy parameters from old conv
    old_conv = C.logging.graph.find_by_name(C.load_model(base_model_path), after_feature_node_name)
    old_param = [p.value for p in old_conv.parameters]
    old_param[0] = np.concatenate([old_param[0]] * (num_inputs - 1),
                                  axis=1)  # need to change the shape
    for i in range(len(new_conv.parameters)):
        print(i, new_conv.parameters[i].shape, old_param[i].shape)
        new_conv.parameters[i].value = old_param[i]

    # Add new dense layer for class prediction
    cloned_out = cloned_layers(new_conv)
    z = C.layers.Dense(num_classes, activation=None, name='fc{}'.format(num_classes))(cloned_out)

    return mc.networkDict(z, label_var, inputs)


def create_model(args):
    stream_func = stream_options[args['stream_type']]
    func_args_name = stream_func.__code__.co_varnames[:stream_func.__code__.co_argcount]

    wanted_args = {
        key: item for key,
        item in args.items() if (key in func_args_name) and (item is not None)
    }
    network = stream_func(**wanted_args)

    mc.print_layers(network['model'])
    output_path = os.path.join(args['outputdir'], args['newNetworkName'])
    mc.save_model(stream_func, wanted_args, network['model'], output_path)


if __name__ == '__main__':
    stream_options = {'OF': get_OF_from_scratch,
                      'RGB': get_RGB_fine_tuning,
                      'RGBdiff': get_RGBdiff_fine_tuning}

    """ Parses input and creates an model."""
    # Adding model arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-stream_type',
                        help='Type of model to be created',
                        choices=stream_options.keys(),
                        required=False,
                        default='OF')
    parser.add_argument('-outputdir',
                        help='Directory where the model will be saved',
                        required=False, default=os.getcwd())
    parser.add_argument('-base_model_path',
                        help='Path to the base model.',
                        required=False,
                        default=None)
    parser.add_argument('-feature_node_name',
                        help='Name of the first node name of the base model to be cloned',
                        required=False,
                        default=None)
    parser.add_argument('-last_node_name',
                        help='Name of the last node name of the base model to be cloned',
                        required=False,
                        default=None)
    parser.add_argument('-num_channels',
                        help='Number of channels in the image input.',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('-image_height',
                        help='Height of the image input.',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('-image_width',
                        help='Width of the image input.',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('-num_classes',
                        help='Number of classes in the model.',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('-num_inputs',
                        help='Number of inputs in the model.',
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument('-dropout',
                        help='Dropout value to be used in the model.',
                        type=float,
                        required=False,
                        default=None)
    parser.add_argument('-newNetworkName',
                        help='Name of the new network.',
                        required=False,
                        default='default_model_name')
    parser.add_argument('-model_name',
                        help='Name of the model type.',
                        required=False,
                        default=None)
    args = vars(parser.parse_args())

    create_model(args)
