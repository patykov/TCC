import numpy as np

def random_crop(img, size=224):
    _, h, w = img.shape
    if (h<size) or (w<size):
        raise ValueError("Image shape: (%d, %d) but crop size: %d"%(h, w, size))
    hh = np.random.randint(h-size)
    ww = np.random.randint(w-size)
    return img[:, hh:hh+size, ww:ww+size]

def center_crop(img, size=224):
    _, h, w = img.shape
    if (h<size) or (w<size):
        raise ValueError("Image shape: (%d, %d) but crop size: %d"%(h, w, size))
    hh = (h-size)//2
    ww = (w-size)//2
    return img[:, hh:hh+size, ww:ww+size]

def random_h_flip(img):
    if np.random.rand() > 0.5:
        return img[:, :, ::-1]
    return img
	
def reescale(img):
	return img*(40.0/255.0) - 40.0/2
	
def reduce_mean(img):
	img -= np.mean(img, axis=(1,2),  keepdims=True)
	return img