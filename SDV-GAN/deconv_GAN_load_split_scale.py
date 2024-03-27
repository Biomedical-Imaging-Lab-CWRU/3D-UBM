# load, split and scale the ubm dataset ready for training
# Running the example loads all images in the training dataset,
# summarizes their shape to ensure the images were loaded correctly,
# then saves the arrays to a new file called ubm_256.npz in compressed NumPy array format

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(512,1024)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		if 'Thumbs.db' not in filename:
			pixels = load_img(path + filename, target_size=size)
			# convert to numpy array
			pixels = img_to_array(pixels)
			# split into satellite and map
			sat_img, map_img = pixels[:, :512], pixels[:, 512:]
			src_list.append(sat_img)
			tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = '../data/ubm/val/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = '../data/ubm/ubm_512_val.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
