import argparse
import os
from stream_training import stream_train


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False)
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
	parser.add_argument('-modelconfig', help="Path to the model configuration file")
	parser.add_argument('-modelname', help="Name of the new model", required=False, default="default_name")
	parser.add_argument('-mapdir', help="Directory with mapfiles")
	parser.add_argument('-stream', help="Stream type to be trained")
	args = parser.parse_args()
		
	# Paths
	data_dir = args.datadir
	newModelName   = args.modelname
	logFile = args.logdir
	map_dir = os.path.join(data_dir, args.mapdir)
	output_dir = os.path.join(args.outputdir, newModelName)
	train_mapFiles = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f]
	new_model_file = os.path.join(output_dir, newModelName+'.model')
	
	stream_train(args.stream, train_mapFiles, output_dir, logFile, newModelName)
	
	C.Communicator.finalize()