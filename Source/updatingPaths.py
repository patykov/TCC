import sys
import os


if __name__ == '__main__':
    datasetDir = sys.argv[1]

    files = os.listdir(datasetDir)
    for file in files:
		filePath = os.path.join(datasetDir, file)
		with open(filePath, 'r') as file:
			lines = file.readlines()
		lines = [l.replace('E:', 'F:') for l in lines]
		with open(filePath, 'w') as file:
			for l in lines:
				file.write(l)