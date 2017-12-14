import argparse
import cntk as C
from cntk.train.training_session import *
import tables
import img_transforms as t
from models import *
import numpy as np
from scipy import sparse as sp
import os


class OFDeserializer(C.io.UserDeserializer):
	def __init__(self, filename, streams, transforms=[], chunksize=4*224*224*20):
		super(OFDeserializer, self).__init__()
		self._chunksize = chunksize
		self._filename = filename
		self._transforms = transforms
		
		# Create the information about streams based on the user provided data
		self._streams = streams

		hdf5_file = tables.open_file(self._filename,  mode='r', root_uep="/labels")
        self.video_labels = np.array(hdf5_file.root.labels)
        hdf5_file.close()
        self._num_chunks = self.video_labels.shape[0]
		
	def stream_infos(self):
		return self._streams

	def num_chunks(self):
		return self._num_chunks

	# Ok, let's actually get the work done
	def get_chunk(self, chunk_id):
		# print("Getting chunk {}".format(chunk_id))
		hdf5_file = tables.open_file(self._filename,  mode='r', root_uep="/video%d"%chunk_id)
        video_u = hdf5_file.root.frames[0]
        video_v = hdf5_file.root.frames[1]
        hdf5_file.close()
        
        max_size = video_u.shape[0] - 9
        frame_id = np.random.randint(max_size)
        sample = np.concatenate((video_u[frame_id:frame_id+10], video_v[frame_id:frame_id+10]))
        
        label = self.video_labels[chunk_id]
			
		for transform in self._transforms:
			sample = transform(sample)
			
		oneHot = np.zeros(101)
		oneHot[label-1] = 1.0
		result = {}
		result[self._streams[0].m_name] = np.ascontiguousarray([sample], dtype=np.float32)
		result[self._streams[1].m_name] = sp.csr_matrix([oneHot], dtype=np.float32)
		return result

		
# Create OF network model by adding a pre input
def create_OF_model(num_classes, num_channels, image_height, image_width):
	input_var = C.input_variable((num_channels, image_height, image_width), name='features')
	label_var = C.input_variable(num_classes, name='label')

	# Create model
	z = create_vgg16(input_var, num_classes, 0.9)

	# Loss and metric
	ce = C.losses.cross_entropy_with_softmax(z, label_var)
	pe = C.metrics.classification_error(z, label_var)
	
	return dict({
		'model': z,
		'ce' : ce,
		'pe' : pe,
		'label': label_var,
		'features': input_var})

# Trains a transfer learning model
def train_model(train_map_file, output_dir, log_file, model_name, profiling=False):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	stack_length = 10
	num_classes	 = 101

	with h5py.File(train_map_file, 'r') as f:
		epoch_size = f["labels"].shape[0]

	# Learning parameters
	iterations	  = 80000*256
	max_epochs	  = round(iterations/epoch_size)
	mb_size		  = 32 #256
	lr_per_mb	  = [0.01]*50000 + [0.001]*20000 + [0.0001]
	l2_reg_weight = 0.0001
	
	print('Epoch_size: {} max_epochs: {}'.format(epoch_size, max_epochs))
	
	# Create network:
	network = create_OF_model(num_classes, 2*stack_length, image_height, image_width)

	# Set learning parameters
	lr_schedule = C.learners.learning_rate_schedule(lr_per_mb, unit=C.UnitType.minibatch)
	mm_schedule = C.learners.momentum_schedule(0.9)

	# Printer
	progress_printer = C.logging.ProgressPrinter(freq=10, tag='Training', log_to_file=log_file, 
									  num_epochs=max_epochs, gen_heartbeat=True)
									  # rank=C.Communicator.rank())

	# Trainer object
	local_learner = C.learners.momentum_sgd(network['model'].parameters, lr_schedule, mm_schedule, 
								unit_gain=False, l2_regularization_weight = l2_reg_weight)
	# learner = C.data_parallel_distributed_learner(local_learner, 
												# num_quantization_bits = 32, 
												# distributed_after = 0)
	learner = local_learner
	trainer = C.Trainer(network['model'], (network['ce'], network['pe']), learner, progress_printer)

	# Create train reader:
	transforms = [t.random_crop, t.random_h_flip, t.reescale, t.reduce_mean]
	streams = [C.io.StreamInformation('image', 0, 'dense', np.float32, shape=(20,224,224,)),
				C.io.StreamInformation('label', 1, 'sparse', np.float32, shape=(101,))]
	my_reader = OFDeserializer(train_map_file, streams, transforms)
	train_reader = C.io.MinibatchSource([my_reader], randomize=True, max_sweeps=max_epochs)

	# define mapping from intput streams to network inputs
	input_map = {
		network['label']: train_reader.streams.label,
		network['features']: train_reader.streams.image
	}

	if profiling:
		C.debugging.start_profiler(dir=output_dir, sync_gpu=True)

	# for epoch in range(max_epochs):		  # loop over epochs
		# sample_count = 0
		# while sample_count < epoch_size:  # loop over minibatches in the epoch
			# data = train_reader.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
			# print(data)
			# trainer.train_minibatch(data)							          # update model with it
			# sample_count +=  data[network['label']].num_samples  # count samples processed so far
			# if sample_count % (100 * mb_size) == 0:
				# print ("Processed {0} samples".format(sample_count))

		# trainer.summarize_training_progress()
		
	training_session(
		trainer=trainer, mb_source=train_reader,
		model_inputs_to_streams=input_map,
		mb_size=mb_size,
		progress_frequency=epoch_size,
		checkpoint_config=CheckpointConfig(filename=os.path.join(output_dir, model_name), restore=True,
											frequency=int(epoch_size/2))
	).train()
	trainer.save_checkpoint(os.path.join(output_dir, "{}_last.dnn".format(model_name)))
	
	if profiling:
		C.debugging.stop_profiler()
	
	return network['model']
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default='/hdfs/pnrsy/t-pakova')
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
	args = parser.parse_args()
	
	C.device.try_set_default_device(C.device.gpu(0))
	
	# Paths
	newModelName   = "VGG16_ofDeserializer"
	train_map_file = os.path.join(args.datadir, "ucf101_of_train_tables.h5")
	new_model_file = os.path.join(args.outputdir, newModelName+'.dnn')
	log_file = os.path.join(args.logdir, 'log.txt')
	
	if not os.path.exists(args.outputdir):
		os.mkdir(args.outputdir)
	
	trained_model = train_model(train_map_file, args.outputdir, log_file, newModelName)
	# trained_model.save(new_model_file)

	print("Stored trained model at %s" % new_model_file)
	
	C.Communicator.finalize()