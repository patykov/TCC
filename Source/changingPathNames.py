import sys
import os
import numpy as np


if __name__ == '__main__':
	dirName = "F:/TCC/Datasets/OF_mapFiles_forPhilly"
	newDirName = "F:/TCC/Datasets/OF_mapFiles_divided"
	
	if not os.path.exists(newDirName):
		os.mkdir(newDirName)
	
	numberOfDivs = 20
	files = os.listdir(dirName)
	doneFiles = os.listdir(newDirName)
	files = [f for f in files if f not in doneFiles]
	for filePath in files:
		with open(os.path.join(dirName, filePath), 'r') as file:
			lines = file.readlines()
		# if "v.zip@/" in lines[0]:
			# newLines = [l.replace("v.zip@/", "v.zip@/v/") for l in lines]
		# else:
			# newLines = [l.replace("u.zip@/", "u.zip@/u/") for l in lines]
		# with open(os.path.join(newDirName, filePath), 'w') as file:
			# for l in newLines:
				# file.write(l)
		if 'test' in filePath:
			with open(os.path.join(newDirName, filePath), 'w') as file:
				for l in lines:
					file.write(l)
		else:
			parts = int(len(lines)/numberOfDivs)
			subfiles = np.split(np.array(lines), [parts*(i+1) for i in range(numberOfDivs-1)])
			print([parts*(i+1) for i in range(numberOfDivs)])
			print(np.array(subfiles).shape)
			for i, f in enumerate(subfiles):
				with open(os.path.join(newDirName, filePath.replace('.txt', 'part{}.txt'.format(i))), 'w') as file:
					for l in f:
						file.write(l)
			