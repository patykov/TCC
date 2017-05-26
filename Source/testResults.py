import sys
import os

if __name__ == '__main__':
	filePath = sys.argv[1]
	right = dict()
	wrong = dict()
	numRight = 0
	numWrong = 0
	total = 0
	rightConfidence = 0
	wrongConfidence = 0
	with open(filePath, 'r') as file:
		for line in file:
			total+=1
			a = line.split(' ')
			rightLabel = int(a[2].replace(':Predicted', ''))
			predicted = int(a[4].replace(',',''))
			confidence = float(a[6].replace('%',''))
			if rightLabel == predicted:
				numRight+=1
				rightConfidence+=confidence
				if rightLabel in right.keys():
					right[rightLabel]+=1
				else:
					right[rightLabel]=1
			else:
				numWrong+=1
				wrongConfidence+=confidence
				if rightLabel in right.keys():
					wrong[rightLabel]+=1
				else:
					wrong[rightLabel]=1

	outputPath = os.path.join(os.basedir(filePath), 'testResults.txt')
	with open(outputPath, 'w') as file:
		file.write('{:<30} {:<5} | {:.2f}% | {:<30} {:.2f}%\n'.format('Total of right: ', numRight, float(numRight)/total), 'With confidence: ', float(rightConfidence)/numRight)
		file.write('{:<30} {:<5} | {:.2f}% | {:<30} {:.2f}%\n'.format('Total of wrong: ', numWrong, float(numWrong)/total), 'With confidence: ', float(wrongConfidence)/numWrong)
		file.write(right)
		file.write('\n')
		file.write(wrong)


