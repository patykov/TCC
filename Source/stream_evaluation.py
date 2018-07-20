import argparse
import os
import numpy as np
import cntk as C

import readers as r


# Get the video label based on its frames evaluations
def getFinalLabel(predictedLabels, labelsConfidence):
    maxCount = max(predictedLabels.values())
    top_labels = [label for label in predictedLabels.keys() if predictedLabels[label] == maxCount]
    # Only one label, return it
    if len(top_labels) == 1:
        confidence = labelsConfidence[top_labels[0]] / maxCount
    # 2 or more labels, need to check confidence
    else:
        topConfidence = dict()
        for label in top_labels:
            topConfidence[label] = labelsConfidence[label] / maxCount
        confidence = max(topConfidence.values())
        top_labels = [label for label in topConfidence.keys() if topConfidence[label] == confidence]
    return top_labels[0], confidence * 100


def eval_and_write(network, test_mapFiles, output_file, stream_type):
    num_classes = 101

    if stream_type == 'of':
        num_inputs = 20
    elif stream_type == 'rgb':
        num_inputs = 1
    elif stream_type == 'rgbdiff':
        num_inputs = 5
    else:
        raise Exception("Stream type unknown!")

    # Create test reader
    transform = r.get_crop_transfrom(crop_type='center')
    test_reader = r.create_mb_source(map_files=test_mapFiles, num_inputs=num_inputs,
                                     transforms=transform, num_classes=num_classes,
                                     max_epochs=1, randomize=True)

    input_map = {}
    if num_inputs == 1:
        input_map[network.find_by_name("data")] = test_reader.streams["feature0"]
    else:
        for i in range(num_inputs):
            input_map[
                network.find_by_name("input_" + str(i))] = test_reader.streams["feature" + str(i)]

    with open(test_mapFiles[0], 'r') as file:
        lines = file.readlines()
    max_samples = len(lines)

    correctLabels = [0] * int(max_samples / 25)
    for i in range(int(max_samples / 25)):
        label = lines[i * 25].replace('\n', '').split('\t')[-1]
        correctLabels[i] = int(label)

    sample_count = 0.0
    results = '{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence')
    while sample_count < max_samples:
        mb = test_reader.next_minibatch(25, input_map=input_map)
        predictedLabels = dict((key, 0) for key in range(num_classes))
        labelsConfidence = dict((key, 0) for key in range(num_classes))
        id_correctLabel = int(sample_count / 25)
        sample_count += 25
        output = network.eval(mb)
        predictions = C.softmax(np.squeeze(output)).eval()
        top_classes = [np.argmax(p) for p in predictions]
        for i, c in enumerate(top_classes):
            predictedLabels[c] += 1  # Melhorar
            labelsConfidence[c] += predictions[i][c]
        label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
        results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(
            correctLabels[id_correctLabel], label, confidence)
        if sample_count % 100 == 0:
            print("{:.2f}".format(float(sample_count) / max_samples))
    with open(output_file, 'w') as file:
        file.write(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',
                        help="Data directory",
                        required=False,
                        default="E:/TCC/Datasets")
    parser.add_argument('-output_dir',
                        help='Output directory',
                        required=False,
                        default="E:/TCC/Results")
    parser.add_argument('-model_path',
                        help="Path to the model to be evaluated",
                        required=True)
    parser.add_argument('-map_dir',
                        help="Directory with mapfiles relative to datadir",
                        required=True)
    parser.add_argument('-stream_type',
                        help="Stream type to be trained.",
                        required=True)
    args = parser.parse_args()

    # Paths
    data_dir = args.data_dir
    output_dir = args.output_dir
    map_dir = os.path.join(data_dir, args.map_dir)
    stream_type = args.stream_type
    test_mapFiles = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f])
    model_name = args.model_path.split("/")[-1].split(".")[0]
    print(model_name)
    output_file = os.path.join(output_dir, "eval_{}.txt".format(model_name))

    # Load Model
    trained_model = C.load_model(args.model_path)
    trained_model = C.combine([trained_model.outputs[0].owner])

    # evaluate model and write out the desired output
    eval_and_write(trained_model, test_mapFiles, output_file, args.stream_type)

    print("Done. Wrote output to %s" % output_file)
