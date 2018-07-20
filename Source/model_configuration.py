import os
import json
import inspect
import cntk as C


def clone_layers(base_model_path, feature_node_name, last_node_name, freeze):
    base_model = C.load_model(base_model_path)
    feature_node = C.logging.graph.find_by_name(base_model, feature_node_name)
    last_node = C.logging.graph.find_by_name(base_model, last_node_name)

    # Clone the desired layers
    cloned_layers = C.combine([last_node.owner]).clone(
        C.CloneMethod.freeze if freeze else C.CloneMethod.share,
        {feature_node: C.placeholder(name='features')})

    return cloned_layers


def get_multiple_inputs(num_inputs, num_channels, image_height, image_width):
    inputs = []
    for c in range(num_inputs):
        inputs.append(
            C.input_variable(
                (num_channels,
                 image_height,
                 image_width),
                name='input_{}'.format(c)))
    return inputs


def networkDict(z, label_var, inputs):
    # Loss and metric
    ce = C.losses.cross_entropy_with_softmax(z, label_var)
    pe = C.metrics.classification_error(z, label_var)

    features = {}
    for i, input in enumerate(inputs):
        features['feature' + str(i)] = input

    return dict({
        'model': z,
        'ce': ce,
        'pe': pe,
        'label': label_var
    }, **features)


def save_model(stream_func, wanted_args, model, output_path):
    model_output_file = output_path + '.model'
    config_output_file = output_path + '.json'

    func_args, _, _, defaults = inspect.getargspec(stream_func)
    config_data = {a: d for (a, d) in zip(func_args, defaults)}
    config_data.update(wanted_args)
    node_outputs = C.logging.get_node_outputs(model)
    config_data['last_node_name'] = node_outputs[0].name

    with open(config_output_file, 'w') as file:
        json.dump(config_data, file, indent=4)
    model.save(model_output_file)


def fetch_model(model_config_file):
    # Assume model_file and config_model_file have the same name and are in the same dir
    model_file = model_config_file.replace('json', 'model')
    if not os.path.isfile(model_file):
        model_file = model_config_file.replace('json', 'dnn')
    z = C.Function.load(model_file)

    # Loading model and ancillary data
    with open(model_config_file) as json_file:
        base_model_data = json.load(json_file)

    return networkDict(z, z.outputs[0], z.arguments), base_model_data


def print_layers(z):
    node_outputs = C.logging.get_node_outputs(z)
    for l in node_outputs:
        print("	{0} {1}".format(l.name, l.shape))
