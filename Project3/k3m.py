import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from tqdm import tqdm

from kmm import neighbour_weight, get_neighbours, get_global_thresholding, otsu_threshold
from lookup_tables import PHASES, A0, A1pix


def iterative(result, changed):
    for n in PHASES.keys():
        result, changed = phase(result, n, changed)
    for (i, j), v in np.ndenumerate(result):
        if result[i, j] == 2:
            result[i, j] = 1
    return result, changed


def phase(img, n, changed):
    result = np.copy(img)
    for (i, j), v in np.ndenumerate(img):
        try:
            if n == 0 and neighbour_weight(get_neighbours(img, i, j)) in A0:
                result[i, j] = 2
            elif result[i, j] == 2:
                if neighbour_weight(get_neighbours(result, i, j)) in PHASES[n]:
                    changed = True
                    result[i, j] = 0
        except IndexError:
            pass
    return result, changed


def k3m(img):
    result = np.copy(img)
    threshold = otsu_threshold(result)
    result = get_global_thresholding(result, threshold)
    result_changed = False
    result, result_changed = iterative(result, result_changed)
    for _ in tqdm(range(10)):
        result_changed = False
        result, result_changed = iterative(result, result_changed)
        if not result_changed:
            break
    for (i, j), v in np.ndenumerate(result):
        try:
            if result[i, j] != 0 and neighbour_weight(get_neighbours(result, i, j)) in A1pix:
                result[i, j] = 0
        except IndexError:
            pass
    return result


if __name__ == '__main__':
    filename = '../data/images/signatures/1.jpg'
    img = np.asarray(Image.open(filename))
    result = k3m(img)
    imshow(result, cmap=plt.cm.binary)
    plt.show()
