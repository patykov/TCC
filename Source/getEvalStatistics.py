import argparse
import os


def get_statistics(filePath, outputDir):
    right = dict()
    wrong = dict()
    numRight = 0
    numWrong = 0
    total = 0
    with open(filePath, 'r') as file:
        file.readline()
        for line in file:
            total += 1
            a = [v for v in line.replace('\n', '').split(
                ' ') if v != '' and v != '|' and v != '%\n']
            if len(a) > 1:  # When I want to eval an file while it's being updated
                rightLabel = int(a[0])
                predicted = int(a[1])
                if rightLabel == predicted:
                    numRight += 1
                    if rightLabel in right.keys():
                        right[rightLabel] += 1
                    else:
                        right[rightLabel] = 1
                else:
                    numWrong += 1
                    # wrongConfidence+=confidence
                    if rightLabel in wrong.keys():
                        wrong[rightLabel] += 1
                    else:
                        wrong[rightLabel] = 1

    fileName = filePath.split('\\')[-1].split('.')[0]
    outputPath = os.path.join(outputDir, '{}_statistics.txt'.format(fileName))
    with open(outputPath, 'w') as file:
        file.write('{:<20} {:<8}\n'.format('Dataset size: ', numRight + numWrong))
        file.write(
            '{:<20} {:<8} | {:<5.2f}%\n'.format(
                'Total of right: ',
                numRight,
                float(numRight) *
                100 /
                total))
        file.write(
            '{:<20} {:<8} | {:<5.2f}%\n'.format(
                'Total of wrong: ',
                numWrong,
                float(numWrong) *
                100 /
                total))
        file.write('\nPrecision per class:\n')
        for k, v in right.items():
            precision = (float(v) * 100) / (v + wrong[k] if k in wrong.keys() else v)
            file.write('{:<3} ----> {:<5.2f}%\n'.format(k, precision))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_path',
                        help="Full path to the evaluation file",
                        required=True)
    parser.add_argument('-output_dir',
                        help='Output directory',
                        required=False,
                        default="E:/TCC/Results")
    args = parser.parse_args()

    get_statistics(args.file_path, args.output_dir)
