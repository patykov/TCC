import os
import numpy as np
from scipy.misc import imread
import h5py
import time

if __name__ == '__main__':

	trainListPath = "E:/TCC/Datasets/UCF-101_splits/trainlist01.txt"
	outputFile = "E:/TCC/Datasets/ucf101_of_train_medium2_lzf.h5"
	dataPath = "E:/TCC/Datasets/UCF-101_opticalFlow"

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