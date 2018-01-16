import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from lookup_tables import NEIGHBOURS_ARRAY, KMM_LOOKUP_ARRAY


def check_neighbours(neighbours):
    w = neighbour_weight(neighbours)
    return w in NEIGHBOURS_ARRAY


def get_neighbours(img, i, j):
    return img[i - 1:i + 2, j - 1:j + 2]


def first_marks(img):
    result = np.ndarray(img.shape)
    for (i, j), v in np.ndenumerate(img):
        if img[i, j] == 1:
            result[i, j] = 1
            neighbours = get_neighbours(img, i, j)
            try:
                if neighbours[0][1] == 0 or neighbours[1][0] == 0 or neighbours[1][2] == 0 or neighbours[2][1] == 0:
                    result[i, j] = 2
                else:
                    if neighbours[0][0] == 0 or neighbours[0][2] == 0 or neighbours[2][0] == 0 or neighbours[2][2] == 0:
                        result[i, j] = 3
            except IndexError:
                pass
        else:
            result[i, j] = 0
    return result


def neighbour_weight(nbs):
    val = 0
    if nbs[0][0] != 0:
        val += 128
    if nbs[0][1] != 0:
        val += 1
    if nbs[0][2] != 0:
        val += 2
    if nbs[1][2] != 0:
        val += 4
    if nbs[2][2] != 0:
        val += 8
    if nbs[2][1] != 0:
        val += 16
    if nbs[2][0] != 0:
        val += 32
    if nbs[1][0] != 0:
        val += 64
    return val


def delete_4s(img):
    result = np.ndarray(img.shape)
    for (i, j), v in np.ndenumerate(img):
        result[i, j] = img[i, j]
        if img[i, j] == 2 or img[i, j] == 3:
            neighbours = get_neighbours(img, i, j)
            try:
                if check_neighbours(neighbours):
                    result[i, j] = 0
            except IndexError:
                pass
    return result


def remove_2or3(img, n):
    result = img
    for (i, j), v in np.ndenumerate(img):
        if result[i, j] == n:
            neighbours = get_neighbours(result, i, j)
            try:
                result[i, j] = 0 if neighbour_weight(neighbours) in KMM_LOOKUP_ARRAY else 1
            except IndexError:
                pass
    return result


def otsu_threshold(im):
    pixel_counts = [np.sum(im == i) for i in range(256)]
    s_max = (0, -10)
    ss = []
    for threshold in range(256):
        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])
        mu_0 = sum([i * pixel_counts[i] for i in range(0, threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0
        # calculate
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)
        if s > s_max[1]:
            s_max = (threshold, s)
    return s_max[0]


def get_global_thresholding(img, threshold):
    # http://www.cse.iitd.ernet.in/~pkalra/col783/Thresholding.pdf
    if img.ndim == 2:
        img = np.asarray(Image.fromarray(np.uint8(img * 255), 'L'))
    else:
        img = to_grayscale(img)
    return np.apply_along_axis(_threshold, -1, img, threshold)


def to_grayscale(im, weights=np.c_[0.2989, 0.5870, 0.1140]):
    tile = np.tile(weights, reps=(im.shape[0], im.shape[1], 1))
    return np.sum(tile * im, axis=2)


def _threshold(pixel, threshold):
    return [1. if color > threshold else 0. for color in pixel]


def kmm(img):
    img = np.array(img)
    threshold = otsu_threshold(img)
    binarized = 1-get_global_thresholding(img, threshold)
    fm = first_marks(binarized)
    d = delete_4s(fm)
    for i in [3, 2] * 3:
        d = remove_2or3(d, i)
    return d


if __name__ == '__main__':
    filename = '../data/images/signatures/1.jpg'
    img = np.asarray(Image.open(filename))
    result = kmm(img)
    imshow(result, cmap=plt.cm.binary)
    plt.show()
