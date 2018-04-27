import cntk as C

# Create a minibatch source.
def create_mb_source(map_files, num_inputs, transforms, num_classes, max_epochs, randomize):
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	assert(len(map_files[:num_inputs])==num_inputs)
	
	sources = []
	for i, map_file in enumerate(map_files[:num_inputs]):
		streams = {"feature"+str(i): C.io.StreamDef(field='image', transforms=transforms),
				   "label"+str(i): C.io.StreamDef(field='label', shape=num_classes)}
		sources.append(C.io.ImageDeserializer(map_file, C.io.StreamDefs(**streams)))

	return C.io.MinibatchSource(sources, max_sweeps=max_epochs, randomize=randomize)
	
def get_input_map(network, num_inputs, train_reader):
	input_map = {network['label']: train_reader.streams.label0}
	for i in range(num_inputs):
		input_map[network['feature'+str(i)]] = train_reader.streams["feature"+str(i)]
	
	return input_map
	
def get_default_transforms(num_channels, image_height=224, image_width=224, is_training=True):
	transforms = [C.io.transforms.scale(width=image_width, height=image_height, 
										channels=num_channels, interpolations='linear')]
	
	if is_training:
		transforms += get_crop_transfrom(crop_type='randomside', crop_size=min(image_height, image_width),
										image_height=image_height, image_width=image_width)
	
	else:
		transforms += get_crop_transfrom(crop_type='center', crop_size=min(image_height, image_width))

	return transforms

def get_crop_transfrom(crop_type='randomside', crop_size=224, num_channels=3, 
						image_height=224, image_width=224):
	return [C.io.transforms.crop(crop_type=crop_type, crop_size=crop_size), C.io.transforms.scale(
			width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
	
def get_color_transfrom(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.2):
	return [C.io.transforms.color(brightness_radius=brightness_radius, contrast_radius=contrast_radius, 
					saturation_radius=saturation_radius)]
