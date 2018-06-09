import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def readImg(path):
    # Returns matrix of (width x height x 3)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def readImgGray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def saveImg(img, path, img_type):
    if img.dtype == bool:
        img = img.astype('uint8')*128
    cv2.imwrite(path + "." + img_type, img)


def readMask(path):
    return readImgGray(path) > 0


def min_max(matrix):
    return np.amin(matrix), np.amax(matrix)


def shrink_mask_square(img, range):
    non_zero = np.nonzero(img)
    x_min, x_max = min_max(non_zero[0])
    y_min, y_max = min_max(non_zero[1])
    c = img.shape[0] / 2
    diff = (c-x_min, x_max-c, c-y_min, y_max-c)
    crop = int(round(max(diff)))
    c = int(c)
    return crop_img(img, (c-crop, c+crop, c-crop, c+crop)), range * crop/c


def shrink_mask(mask, extent=None):
    non_zero = np.nonzero(mask)
    x_min, x_max = min_max(non_zero[0])
    y_min, y_max = min_max(non_zero[1])
    idxs = (x_min, x_max, y_min, y_max)
    new_mask = crop_img(mask, idxs)
    if extent is None:
        return new_mask
    else:
        return new_mask, calc_extents(mask, extent, idxs)


def calc_extents(img, ext, idxs):
    x1 = idxs[0]/img.shape[0] * (ext[1] - ext[0]) + ext[0]
    x2 = idxs[1]/img.shape[0] * (ext[1] - ext[0]) + ext[0]
    y1 = idxs[2]/img.shape[1] * (ext[3] - ext[2]) + ext[2]
    y2 = idxs[3]/img.shape[1] * (ext[3] - ext[2]) + ext[2]
    return (x1, x2, y1, y2)


def crop_img(img, idxs):
    new_img = img[idxs[0]:idxs[1] + 1, idxs[2]:idxs[3] + 1]
    return new_img


def get_mask(out_range, resolution, path, in_range):
    mask = readImgGray(path)
    assert mask.shape[0] == mask.shape[1]
    s = out_range / in_range
    c = round(mask.shape[0] / 2)

    r = round(c * s)
    if s < 1:
        maskCrop = mask[c - r:c + r, c - r:c + r]
    else:
        p = r - c
        maskCrop = np.pad(mask, ((p, p), (p, p)), mode='constant', constant_values=np.amax(mask))
    return cv2.resize(maskCrop, (resolution, resolution), interpolation=cv2.INTER_NEAREST) > 0


def create_range_mask(fill_range, out_range, resolution):
    x = np.linspace(-out_range, out_range, resolution, endpoint=True)
    y = np.linspace(-out_range, out_range, resolution, endpoint=True)
    X, Y = np.meshgrid(x, y)
    d = np.sqrt(np.square(X) + np.square(Y))
    return d > fill_range


def show_heatmap(a):
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def print_np(matrix):
    print(str(matrix.shape) + " - "+ str(matrix.dtype) + " - [" + str(np.amin(matrix)) +", "+ str(np.amax(matrix)) + "]")


def excepthook(type, value, tback):
    # log the exception here
    # then call the default handler
    sys.__excepthook__(type, value, tback)

def set_excepthook():
    sys.excepthook = excepthook  # Traceback when using pyQT.