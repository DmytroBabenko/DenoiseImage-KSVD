import numpy as np

from skimage import io, util
from sklearn.feature_extraction import image

from ksvd import KSVD


def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img

class DenoiseImage:
    def __init__(self):
        self.patch_size = (5, 5)
        self.optimal_fit_size = 10000


    def denoise(self, image_file, out_image_file):
        img = util.img_as_float(io.imread(image_file))
        patches = image.extract_patches_2d(img, self.patch_size)
        signals = patches.reshape(patches.shape[0], -1)
        mean = np.mean(signals, axis=1)[:, np.newaxis]
        signals -= mean


        ksvd = KSVD(k_atoms=32, num_iterations=10, tolerance=0.000001)
        D, X = ksvd.run(signals[:self.optimal_fit_size].T)

        X = ksvd.sparse_coding(D, signals.T)

        reduced = (D.dot(X)).T + mean
        reduced_img = image.reconstruct_from_patches_2d(reduced.reshape(patches.shape), img.shape)
        io.imsave(out_image_file, clip(reduced_img))

denoise_image = DenoiseImage()
denoise_image.denoise("coffee_noisy.png", "coffee_reduced.png")



