# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
from cntk import load_model
from cntk.ops import combine, input_variable, softmax
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs


def create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, map_file):
    transforms = [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
                    ImageDeserializer.mean(mean_file)]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),       # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_output_classes))),    # and second as 'label'.
        randomize=False)


def eval_and_write(model_file, output_file, minibatch_source, input_map, epoch_size):
    # load model
    loaded_model  = load_model(model_file)
    loaded_model  = combine([loaded_model.outputs[2].owner])

    # evaluate model and get desired node output
    print("Evaluating model")
    features_si = minibatch_source['features']
    sample_count = 0
    with open(output_file, 'w') as results_file:
        while sample_count < epoch_size:
            mb = minibatch_source.next_minibatch(1)
            output = loaded_model.eval({loaded_model.arguments[0]:mb[features_si]})
            sample_count += mb[features_si].num_samples
            predictions = softmax(np.squeeze(output)).eval()
            top_class = np.argmax(predictions)
            print("Label: {}, Confidence: {:.2f}%".format(top_class, predictions[top_class] * 100))
            results_file.write('{}\n'.format(top_class))

if __name__ == '__main__':
    # define location of model and data and check existence
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_file  = os.path.join(base_folder, "..", "Models", "CNTK", "ResNet_34.model")
    map_file    = os.path.join(base_folder, "..", "DataSets", "val_map.txt")
    mean_file   = os.path.join(base_folder, "..", "DataSets", "ImageNet1K_mean.xml")

    # create minibatch source
    image_height = 224
    image_width  = 224
    num_channels = 3
    num_output_classes = 1000
    minibatch_source = create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, map_file)

    # # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width), np.float32)
    label_var = input_variable(num_output_classes, np.float32)

    input_map = {
        input_var : minibatch_source.streams.features,
        label_var : minibatch_source.streams.labels
    }

    output_file = os.path.join(base_folder, "predOutput.txt")
    # evaluate model and write out the desired output
    eval_and_write(model_file, output_file, minibatch_source, input_map, epoch_size=8)
    print("Done. Wrote output to %s" % output_file)