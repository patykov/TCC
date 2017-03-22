# import cntk

# loaded_model = cntk.ops.functions.load_model('E:/TCC/Models/CNTK/ResNet_34.model')
# if is_BrainScript: 
#     loaded_model = cntk.ops.combine([loaded_model.outputs[0]])

# parameters = loaded_model.parameters
# for parameter in parameters:
#     print(parameter.name, parameter.shape, "\n", parameter.value)

import cntk
from PIL import Image 
import numpy as np

z = cntk.ops.functions.load_model('E:/TCC/Models/CNTK/ResNet_34.model')
pic = np.array(Image.open("E:/TCC/Source/0_0.jpg"), dtype=np.float32) - 128
pic = np.ascontiguousarray(np.transpose(pic, (2, 0, 1)))
z_out = cntk.combine([z.outputs[2].owner])
predictions = np.squeeze(z_out.eval({z_out.arguments[0]:pic}))
top_class = np.argmax(predictions)