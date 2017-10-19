import sys
import os
import numpy as np


if __name__ == '__main__':
	dirName = "F:/TCC/Datasets/RGB_mapFiles_forLaterXforms"
	newDirName = "F:/TCC/Datasets/RGB_mapFiles"
	
	if not os.path.exists(newDirName):
		os.mkdir(newDirName)
	
	# numberOfDivs = 50
	files = os.listdir(dirName)
	doneFiles = os.listdir(newDirName)
	files = [f for f in files if f not in doneFiles]
	for filePath in files:
		with open(os.path.join(dirName, filePath), 'r') as file:
			lines = file.readlines()
		# if "F:/TCC/Datasets/UCF-101_opticalFlow\\u\\" in lines[0]:
			# newLines = [l.replace("F:/TCC/Datasets/UCF-101_opticalFlow\\u\\", 
						# "/hdfs/pnrsy/t-pakova/u.zip@/u/") for l in lines]
		# else:
			# newLines = [l.replace("F:/TCC/Datasets/UCF-101_opticalFlow\\v\\", 
						# "/hdfs/pnrsy/t-pakova/v.zip@/v/") for l in lines]
						
		newLines = [l.replace("E:\\TCC\\Datasets\\UCF-101_rgb\\", 
							"/hdfs/pnrsy/t-pakova/UCF-101_rgb.zip@/UCF-101_rgb/") for l in lines]
		newLines = [l.replace("F:/TCC/Datasets/UCF-101_rgb\\", 
					"/hdfs/pnrsy/t-pakova/UCF-101_rgb.zip@/UCF-101_rgb/") for l in newLines]
		with open(os.path.join(newDirName, filePath), 'w') as file:
			for l in newLines:
				file.write(l)
				
		# if 'test' in filePath:
			# with open(os.path.join(newDirName, filePath), 'w') as file:
				# for l in newLines:
					# file.write(l)
		# else:
			# parts = int(len(lines)/numberOfDivs)
			# subfiles = np.split(np.array(lines), [parts*(i+1) for i in range(numberOfDivs-1)])
			# print([parts*(i+1) for i in range(numberOfDivs)])
			# print(np.array(subfiles).shape)
			# for i, f in enumerate(subfiles):
				# with open(os.path.join(newDirName, filePath.replace('.txt', 'part{}.txt'.format(i))), 'w') as file:
					# for l in f:
						# file.write(l)
			