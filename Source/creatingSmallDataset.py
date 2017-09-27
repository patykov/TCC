import sys
import os
from shutil import copyfile


if __name__ == '__main__':
	mapDir = "F:/TCC/Datasets/OF_mapFiles"
	newMapDir = "F:/TCC/Datasets/OF_mapFiles_note"
	noteDataDir = "C:/Users/t-pakova/OneDrive - Microsoft/TCC/Datasets/smallOF"
	newDataDir = "F:/TCC/Datasets/UCF-101_OF_small"
	
	if not os.path.exists(newMapDir):
		os.mkdir(newMapDir)
	if not os.path.exists(newDataDir):
		os.mkdir(newDataDir)
	if not os.path.exists(os.path.join(newDataDir, 'u')):
		os.mkdir(os.path.join(newDataDir, 'u'))
	if not os.path.exists(os.path.join(newDataDir, 'v')):
		os.mkdir(os.path.join(newDataDir, 'v'))
	
	files = os.listdir(mapDir)
	for filePath in files:
		with open(os.path.join(mapDir, filePath), 'r') as file:
			lines = file.readlines()
		newLines = [l.replace("F:/TCC/Datasets/UCF-101_opticalFlow", noteDataDir) for l in lines[:102]]
		with open(os.path.join(newMapDir, filePath), 'w') as file:
			for l in newLines:
				file.write(l)
		
		frames = [l.split('\t')[1] for l in lines[:102]]
		for f in frames:
			new_f = f.replace("UCF-101_opticalFlow", "UCF-101_OF_small")
			c = f.split('\\')[2]
			if not os.path.exists(os.path.join(newDataDir, 'v', c)):
				os.mkdir(os.path.join(newDataDir, 'v',c))
			if not os.path.exists(os.path.join(newDataDir, 'u', c)):
				os.mkdir(os.path.join(newDataDir, 'u',c))
			copyfile(f, new_f)
		
		