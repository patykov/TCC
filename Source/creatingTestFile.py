#!/usr/bin/env python

import numpy as np
import sys
import os


if __name__ == '__main__':
	base_folder = 'E:\TCC'
	map_dir = os.path.join(base_folder, 'Datasets', 'TestMapFiles01_RGB')
	new_file = os.path.join(base_folder, 'Datasets', 'TestMapFiles_compare.txt')
	
	frames = []
	map_files = [os.path.join(map_dir, f) for f in sorted(os.listdir(map_dir))]
	selectedFiles = np.random.choice(map_files, 500)
	for file_path in selectedFiles:
		with open(file_path, 'r') as file:
			lines = file.readlines()
		selectedFrames = np.random.choice(lines, 3)
		frames += [f for f in selectedFrames]

	with open(new_file, 'w') as file:
		for f in frames:
			file.write(f)

