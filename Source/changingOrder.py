import sys
import os
from shutil import copyfile


if __name__ == '__main__':
	dirName = "F:/TCC/Datasets/OF_mapFiles-forLaterXforms"
	newDirName = "F:/TCC/Datasets/OF_mapFiles-forLaterXforms_uv"
	
	if not os.path.exists(newDirName):
		os.mkdir(newDirName)
	
	files = os.listdir(dirName)
	for fileName in files:
		if 'test' in fileName:
			# f_id = int(fileName.split('_')[-1].split('.')[0])
			# if f_id%2==0:
				# newId=str(f_id-1)
			# else:
				# newId=str(f_id+1)
			# new_fileName = fileName.split('_')[0]+'_'+newId+'.'+fileName.split('.')[-1]
			new_fileName = fileName
		else:
			f_id = int(fileName.split('_')[-1].split('.')[0])
			if f_id%2==0:
				newId=str(f_id-1)
			else:
				newId=str(f_id+1)
			new_fileName = fileName.split('_')[0]+'_'+newId+'.'+fileName.split('.')[-1]
		copyfile(os.path.join(dirName, fileName), os.path.join(newDirName, new_fileName))