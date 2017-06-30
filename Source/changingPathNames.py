import sys
import os


if __name__ == '__main__':
	dirName = sys.argv[1]
	
	files = os.path.listdir(dirName)
	for filePath in files:
		with open(filePath, 'r') as file:
			lines = file.readlines()
		newLines = [l.replace("E:", "F:") for l in lines]
		with open(filePath, 'w') as file:
			file.write(newLines)