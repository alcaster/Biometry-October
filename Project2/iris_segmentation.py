import numpy as np

from wrappers import return_float32, return_uint8, normalize
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from tqdm import tqdm


def get_negate(img):
    return 1 - img


@return_uint8
def get_grey_image(img):
    return np.apply_along_axis(_pixel2gray, -1, img)


def get_global_thresholding_simple(img, threshold):
    # http://www.cse.iitd.ernet.in/~pkalra/col783/Thresholding.pdf
    img = get_grey_image(img)
    return np.apply_along_axis(_threshold, -1, img, threshold)


@return_uint8
@normalize
def get_contrast(img, contrast: float):
    # http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    F = (259 * (contrast + 255)) / (255 * (259 - contrast))
    return np.apply_along_axis(contrast_pixel, -1, img, F)


@return_float32
def get_dilatation(img, dimension):
    out = np.ndarray(img.shape, dtype=np.uint8)
    for (i, j, c), v in tqdm(np.ndenumerate(img)):
        neighbours = img[i:i + dimension, j:j + dimension]
        out[i, j] = img[i, j]
        for (row, column, color), _ in np.ndenumerate(neighbours):
            if neighbours[row, column, 0] == 1.:
                out[i, j] = [1., 1., 1.]
    return out


@return_float32
def get_erosion(img, dimension):
    out = np.ndarray(img.shape, dtype=np.uint8)
    for (i, j, c), v in tqdm(np.ndenumerate(img)):
        neighbours = img[i:i + dimension, j:j + dimension]
        out[i, j] = img[i, j]
        for (row, column, color), _ in np.ndenumerate(neighbours):
            if neighbours[row, column, 0] == 0.:
                out[i, j] = [0., 0., 0.]
    return out


@return_float32
def get_masked_image(img, mask):
    out = np.ones(img.shape, dtype=np.float32)
    mask = mask.astype(bool)
    out[~mask] = ~img[~mask]
    return out


def get_opening(img, dimension):
    out = get_erosion(img, dimension)
    out = get_dilatation(out, dimension)
    return out


def get_closing(img, dimension):
    out = get_dilatation(img, dimension)
    out = get_erosion(out, dimension)
    return out


def merge_masks(mask1, mask2):
    out = np.zeros(mask1.shape)
    out[(mask1 == 1) | (mask2 == 1)] = 1
    return out


def _pixel2gray(pixel):
    grey = int(sum(pixel) // 3)  # Average of 3 colors
    return [grey] * len(pixel)


def _threshold(pixel, threshold):
    return [1. if color > threshold else 0. for color in pixel]


def contrast_pixel(pixel, factor):
    return [factor * (color - 128) + 128 for color in pixel]


def iris_segmentation(img):
    img = np.asarray(img)
    iris_segmentation = img.copy()
    grey = get_grey_image(iris_segmentation)
    contrast_factor = int(np.sum(grey) / (grey.shape[0] * grey.shape[1] * grey.shape[2]) / 5)

    contrasted = get_contrast(grey, contrast_factor)
    iris_mask = get_global_thresholding_simple(contrasted, 1)
    iris_mask = 1 - iris_mask
    # Morphological operations on iris mask
    iris_mask = get_dilatation(iris_mask, 3)
    iris_mask = get_erosion(iris_mask, 4)
    dilation_times = 2
    for i in range(5, 5 - dilation_times, -1):
        iris_mask = get_dilatation(iris_mask, i)

    # Sclera
    contrasted = get_contrast(grey, -contrast_factor)
    sclera_mask1 = get_global_thresholding_simple(1 - contrasted, 10 * contrast_factor)
    sizes = [2, 3, 5]
    sclera_mask2 = sclera_mask1.copy()
    for i in range(len(sizes)):
        sclera_mask2 = get_dilatation(sclera_mask2, sizes[i])
    sclera_mask2 = 1 - sclera_mask2
    mask_together = merge_masks(iris_mask, sclera_mask2)
    together_color = get_masked_image(img, mask_together)
    return together_color


if __name__ == '__main__':
    filename = '../data/images/eye.jpg'
    img = np.asarray(Image.open(filename))
    segmented = iris_segmentation(img)
    imshow(segmented)
    plt.show()

