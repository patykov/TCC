import sys
import os

if __name__ == '__main__':
	filePath = sys.argv[1]
	right = dict()
	wrong = dict()
	numRight = 0
	numWrong = 0
	total = 0
	with open(filePath, 'r') as file:
		file.readline()
		for line in file:
			total+=1
			a = [v for v in line.replace('\n','').split(' ') if v!='' and v!='|' and v!='%\n']
			if len(a) > 1: # When I want to eval an file while it's being updated
				rightLabel = int(a[0])
				predicted = int(a[1])
				if rightLabel == predicted:
					numRight+=1
					if rightLabel in right.keys():
						right[rightLabel]+=1
					else:
						right[rightLabel]=1
				else:
					numWrong+=1
					if rightLabel in wrong.keys():
						wrong[rightLabel]+=1
					else:
						wrong[rightLabel]=1
		
	outputDir = 'E:/TCC/Results'
	fileName = filePath.split('\\')[-1].split('.')[0]
	outputPath = os.path.join(outputDir, '{}_statistics-perclass.txt'.format(fileName))
	with open(outputPath, 'w') as file:
		precision_per_class = 0
		num_classes = 0
		per_class = ''
		for k, v in right.items():
			precision = (float(v)*100)/(v+wrong[k] if k in wrong.keys() else v)
			precision_per_class += precision
			num_classes +=1
			per_class += '{:<3} ----> {:<5.2f}%\n'.format(k, precision)
		file.write('{:<10} {:<8}\n'.format('Dataset size: ', numRight+numWrong))
		file.write('{:<23} | {:<5.2f}%\n'.format('Total of right per video: ', float(numRight)*100/total))
		file.write('{:<23} | {:<5.2f}%\n'.format('Precision per class: ', float(precision_per_class)/num_classes))
		file.write('\nPrecision per class:\n')
		file.write(per_class)
		
