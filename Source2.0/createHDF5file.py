import argparse
import h5py
import numpy as np
import os
from scipy.misc import imread
import tables
import time


def create_h5py(trainListPath, outputFile, dataPath):
	with open(trainListPath, 'r') as file:
		videos = file.readlines()
	
	start_time = time.time()
	with h5py.File(outputFile, 'w') as f:
		u = f.create_group("u")
		v = f.create_group("v")
		
		# labels = f.create_dataset("labels", (len(videos),), np.uint8)
		labels = f.create_dataset("labels", (100,), np.uint8)
		
		for i, video in enumerate(videos[:100]):
			video_name = video.split('.')[0].split('/')[-1]
			video_label = int(video.split(' ')[-1])
			labels[i] = video_label
			print("Saving video {}, count {}, passed time {:.3f}".format(video_name, i, (time.time()-start_time)))
			
			video_path_u = os.path.join(dataPath, 'u', video_name)
			video_path_v = os.path.join(dataPath, 'v', video_name)
			
			img_list = os.listdir(video_path_u)
			dset_name = "video%d" %i
			dset_size = (len(img_list), 256, 341)
			
			imgs_dset_u = u.create_dataset(dset_name, dset_size, np.uint8, compression='lzf')
			imgs_dset_v = v.create_dataset(dset_name, dset_size, np.uint8, compression='lzf')
			
			for j, img in enumerate(img_list):
				path_u = os.path.join(video_path_u, img)
				path_v = os.path.join(video_path_v, img)
				
				img_u = imread(path_u)
				img_v = imread(path_v)
				
				imgs_dset_u[j] = img_u
				imgs_dset_v[j] = img_v

def create_pytables(trainListPath, outputFile, dataPath):

	with open(trainListPath, 'r') as file:
		videos = file.readlines()
	
	start_time = time.time()
	labels = []
	filters = tables.Filters(complevel=5, complib='zlib')
	hdf5_file = tables.open_file(outputFile, mode='a', filters=filters)

	for i, video in enumerate(videos):
		video_name = video.split('.')[0].split('/')[-1]
		video_label = int(video.split(' ')[-1])
		
		video_path_u = os.path.join(dataPath, 'u', video_name)
		video_path_v = os.path.join(dataPath, 'v', video_name)
		
		img_list = os.listdir(video_path_u)
		dset_name = "video%d" %i
		labels.append(video_label-1)
		
		imgs_u = []
		imgs_v = []
		for j, img in enumerate(img_list):
			path_u = os.path.join(video_path_u, img)
			path_v = os.path.join(video_path_v, img)
			
			imgs_u.append(imread(path_u))
			imgs_v.append(imread(path_v))			

		group = hdf5_file.create_group("/", dset_name)
		data_storage = hdf5_file.create_carray(group, "frames", obj=[imgs_u, imgs_v])

		print("Saving video {}, count {}, passed time {:.3f}".format(video_name, i, (time.time()-start_time)))
				
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-type', help='HDF5 type to use: tables or h5py', required=False, default='tables')
	parser.add_argument('-trainListPath', help='Path to the video map file', required=False, default=None)
	parser.add_argument('-outputFile', help='Output file path', required=False, default=None)
	parser.add_argument('-dataPath', help="Data path", required=False, default=None)
	args = parser.parse_args()
		
	trainListPath = args.trainListPath
	outputFile = args.outputFile
	dataPath = args.dataPath
	
	if trainListPath == None:
		trainListPath = "E:/TCC/Datasets/UCF-101_splits/trainlist01.txt"
	if outputFile == None:
		outputFile = "E:/TCC/Datasets/hdf5_file.h5"
	if dataPath == None:
		dataPath = "E:/TCC/Datasets/UCF-101_opticalFlow"
		
	if args.type == "tables":
		create_pytables(trainListPath, outputFile, dataPath)
	if args.type == "h5py":
		create_h5py(trainListPath, outputFile, dataPath)
	else:
		print("HDF5 type not known.")
	
	
	
	
	
	