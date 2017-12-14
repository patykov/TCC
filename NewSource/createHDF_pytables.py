import os
import numpy as np
from scipy.misc import imread
import tables
import time

if __name__ == '__main__':

	trainListPath = "E:/TCC/Datasets/UCF-101_splits/trainlist01.txt"
	outputFile = "E:/TCC/Datasets/ucf101_of_train_tables.h5"
	dataPath = "E:/TCC/Datasets/UCF-101_opticalFlow"

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
	
	group = hdf5_file.create_group("/", 'labels')
	data_storage = hdf5_file.create_carray(group, 'labels', obj=labels)
	hdf5_file.close()