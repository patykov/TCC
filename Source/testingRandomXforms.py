import numpy as np
import cntk as C
import cntk.io.transforms as xforms 
import sys

def create_reader(map_file1, map_file2):
    transforms = [xforms.crop(crop_type='randomside', crop_size=224),
                  xforms.scale(width=224, height=224, channels=3, interpolations='linear')]
    source1 = C.io.ImageDeserializer(map_file1, C.io.StreamDefs(
        source_image = C.io.StreamDef(field='image', transforms=transforms)))
    source2 = C.io.ImageDeserializer(map_file2, C.io.StreamDefs(
        target_image = C.io.StreamDef(field='image', transforms=transforms)))
    return C.io.MinibatchSource([source1, source2], max_samples=sys.maxsize, randomize=False)

x = C.input_variable((3,224,224))
y = C.input_variable((3,224,224))
z = C.squared_error(x, y)


mapFile = "path_to_file"
reader = create_reader(mapFile, mapFile)

minibatch_size = 3

input_map={
    x: reader.streams.source_image,
    y: reader.streams.target_image
}

for i in range(9):
    data=reader.next_minibatch(minibatch_size, input_map=input_map)
    results = z.eval(data)
    print(results)
    