from __future__ import print_function
import os
import argparse
import math
import numpy as np

from cntk.utils import *
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk import Trainer, cntk_py, load_model, combine
from cntk.device import set_default_device, gpu
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from _cntk_py import set_computation_network_trace_level


# Paths relative to current python file.
abs_path   = "E:\TCC"
data_path  = os.path.join(abs_path, "DataSets")
model_path = os.path.join(abs_path, "Models", "CNTK")

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 101

# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file, train):
    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            ImageDeserializer.crop(crop_type='Random', jitter_type='uniratio') # train uses jitter
        ]
    transforms += [
        ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        ImageDeserializer.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes))))   # and second as 'label'


# Train the network.
def train(reader_train, model_file, epoch_size, max_epochs):

    set_computation_network_trace_level(0)
	
    # create model, and configure learning parameters 
    loaded_model = load_model(model_file)
    z = combine([loaded_model.outputs[2].owner])
    lr_per_mb = [0.01]*4+[0.01]*2
	
    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width), dynamic_axes=z.dynamic_axes, name = 'features')
    label_var = input_variable(num_classes, dynamic_axes=z.dynamic_axes, name = 'labels')

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # shared training parameters 
    minibatch_size = 256
    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    
    # trainer object
    learner     = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                               l2_regularization_weight = l2_reg_weight)
    trainer     = Trainer(z, ce, pe, learner)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)
        z.save_model(os.path.join(model_path, "ResNet34_{}.dnn".format(epoch)))
    

if __name__=='__main__':
    set_default_device(gpu(0))

    mean_data=os.path.join(data_path, 'ImageNet1K_mean.xml')
    output_path=os.path.join(abs_path, 'Output')
    output_file=os.path.join(output_path, 'ResNet_34')
    train_data=os.path.join(data_path, 'train_map.txt')

    model_file = os.path.join(model_path, 'ResNet_34.0')
    epochs = 6
    network_name = 'ResNet_34'
    
    reader_train = create_reader(train_data, mean_data, True)

    epoch_size = 888271
    train(reader_train, model_file, epoch_size, epochs)
