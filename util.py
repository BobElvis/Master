import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path as path
import pickle


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


def show_mask(a, bg=None, show=True):
    pass


def show_heatmap(a, bg=None, alpha_mask=None, show=True, axes=False):
    if bg is None:
        fig = plt.imshow(a, cmap='hot')
        plt.colorbar()
    else:
        cm = plt.get_cmap('hot')
        a_scaled = a.astype(float)/np.amax(a)
        img = cm(a_scaled)
        if alpha_mask is None:
            img[a == 0, 3] = 0
        else:
            img[alpha_mask, 3] = 0
        bg.add_image_alpha(img, bg.radar_range)
        img_out, _ = bg.get_img()
        img_out = crop_img(img_out, (450, 950, 100, 900))
        fig = plt.imshow(img_out)
        sm = plt.cm.ScalarMappable(cmap=cm)
        sm.set_array(a)
        plt.colorbar(sm)
    if show:
        plt.show()
    if not axes:
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


def save_data(data, filename):
    filehandler = open(filename + ".obj", 'wb')
    success = False
    print("Saving...")
    while not success:
        try:
            pickle.dump(data, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
            success = True
        except RecursionError:
            print("Recursion error. max:{}".format(sys.getrecursionlimit()))
            sys.setrecursionlimit(sys.getrecursionlimit() * 2)
    filehandler.close()
    print("Saved successfully to '{}'".format(filename))


def load_data(filename):
    try:
        print("Restoring {}...".format(filename))
        filehandler = open(filename + ".obj", 'rb')
    except FileNotFoundError:
        print("File {} not found..".format(filename))
        raise FileNotFoundError
    return pickle.load(filehandler)


def get_filename(root, basename, newfile, number=None):
    if newfile or number is None:
        import dataset
        files = dataset.find_files(root, extension="obj")
        files = [(f, int(f.split(basename)[1])) for f in files if f.startswith(basename)]
        files_sorted = sorted(files, key=lambda x: x[1])
        top_number = int(files_sorted[-1][1])
        number = top_number + 1 if newfile else top_number
    return path.join(root, "{} {}".format(basename, number))


def print_np(matrix):
    print(str(matrix.shape) + " - "+ str(matrix.dtype) + " - [" + str(np.amin(matrix)) +", "+ str(np.amax(matrix)) + "]")


def excepthook(type, value, tback):
    # log the exception here
    # then call the default handler
    sys.__excepthook__(type, value, tback)

def set_excepthook():
    sys.excepthook = excepthook  # Traceback when using pyQT.