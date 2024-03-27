# example of loading a pix2pix model and using it for UBM enhancement
from keras.models import load_model
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np

import imageio
import time


# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()

def save_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	
	for i in range(len(images)):
		imageio.imwrite("results_ + str(i) + '.png' " , images[i],pilmode="RGB")



# load dataset
[X1, X2] = load_real_samples('../data/ubm/ubm_512_val.npz')
print('Loaded', X1.shape, X2.shape)
# load model
model = load_model('./unet/model_001800.h5')
model.summary()
# select random example
# ix = randint(0, len(X1), 1)
# src_image, tar_image = X1[ix], X2[ix]
# generate image from source
t = time.time()
predictions = model.predict(X1)
elapsed = time.time() - t
print ("Elapsed time: ", elapsed, " seconds")
print("predictions shape:", predictions.shape)

np.save('ubm_512_val_results_unet', predictions)

# save_images(X1, predictions, X2)
# # plot all three images
# plot_images(src_image, gen_image, tar_image)