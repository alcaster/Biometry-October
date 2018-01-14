import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

A0 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60,
      62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
      143, 159, 191, 192, 193, 195, 199, 207, 223, 224,
      225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
      251, 252, 253, 254]
A1 = [7, 14, 28, 56, 112, 131, 193, 224]
A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135,
      193, 195, 224, 225, 240]
A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120,
      124, 131, 135, 143, 193, 195, 199, 224, 225, 227,
      240, 241, 248]
A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
      224, 225, 227, 231, 240, 241, 243, 248, 249, 252]
A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
      124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
      207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
      249, 251, 252, 254]
A1pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56,
         60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131,
         135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
         224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
         249, 251, 252, 253, 254]
PHASES = {
    0: A0,
    1: A1,
    2: A2,
    3: A3,
    4: A4,
    5: A5,
}


def iterative_part(result, changed):
    for n in PHASES.keys():
        result, changed = phase_function(result, n, changed)
    for (i, j), v in np.ndenumerate(result):
        if result[i, j] == 2:
            result[i, j] = 1
    return result, changed


def phase_function(img, n, changed):
    result = np.copy(img)
    for (i, j), v in np.ndenumerate(img):
        try:
            if n == 0 and get_neighbour_weight(img[i - 1:i + 2, j - 1:j + 2]) in A0:
                result[i, j] = 2
            elif result[i, j] == 2:
                if get_neighbour_weight(result[i - 1:i + 2, j - 1:j + 2]) in PHASES[n]:
                    changed = True
                    result[i, j] = 0
        except IndexError:
            pass
    return result, changed


def get_neighbour_weight(neighbours):
    weight = 0
    if neighbours[0][1] != 0:
        weight += 1
    if neighbours[0][2] != 0:
        weight += 2
    if neighbours[1][2] != 0:
        weight += 4
    if neighbours[2][2] != 0:
        weight += 8
    if neighbours[2][1] != 0:
        weight += 16
    if neighbours[2][0] != 0:
        weight += 32
    if neighbours[1][0] != 0:
        weight += 64
    if neighbours[0][0] != 0:
        weight += 128
    return weight


def k3m(img):
    result = np.copy(img)
    result_changed = False
    result, result_changed = iterative_part(result, result_changed)
    i = 0
    while result_changed:
        i += 1
        print(i)
        if i > 100:
            break
        result_changed = False
        result, result_changed = iterative_part(result, result_changed)
    for (i, j), v in np.ndenumerate(result):
        try:
            if result[i, j] != 0 and get_neighbour_weight(result[i - 1:i + 2, j - 1:j + 2]) in A1pix:
                result[i, j] = 0
        except IndexError:
            pass
    return result


if __name__ == '__main__':
    filename = '../data/images/signatures/3.jpg'
    img = np.asarray(Image.open(filename))
    a = k3m(img)
    imshow(a)
    im = Image.fromarray(a)
    im = im.convert('RGB')
    im.save("lol.jpeg")
