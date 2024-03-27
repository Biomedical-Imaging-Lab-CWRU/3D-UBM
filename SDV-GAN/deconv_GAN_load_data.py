# load the prepared dataset
# Running this example loads the prepared dataset
# and summarizes the shape of each array, confirming our expectations
# of a little over one thousand 512×512 image pairs.

from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('../data/ubm/ubm_0.02.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
print('Loaded: ', src_images.shape, tar_images.shape)
# plot source images
n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()