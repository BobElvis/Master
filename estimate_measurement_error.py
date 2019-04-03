import dataset as dataset
import detection.detection as detection
import numpy as np
import util
import cv2
import gc
import time
import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

if __name__ == '__main__':

    create = True
    #folder = '2018-05-31'
    folder = '2018-06-01 2018-06-07'
    load_name = "sizes_{}2.obj".format(folder)

    # SETTINGS:
    dataset = dataset.DatasetFolder(dataset.ROOT_FOLDER, folder)
    detect_data = detection.DetectData(dataset.config)
    dataloader = detection.Detection(dataset, detect_data)
    dataloader.areaMinCnt = 3

    # SETTINGS:
    i_start = 0
    i_end = len(dataset)

    # Go through samples:
    if create:
        t = time.time()
        print("i=({},{}). {} elements in dataset".format(i_start, i_end, len(dataset)))
        sample_indices = range(i_start, i_end)
        n = len(sample_indices)
        for idx, data_idx in enumerate(sample_indices):
            if idx % (n/30) < 1:
                gc.collect()
                print("{}/{}".format(idx, n))
            dataloader.load_data(idx)


        # Saving the avg-img:
        error = dataloader.error
        filehandler = open(load_name, 'wb')
        pickle.dump(dataloader.error, filehandler)
        filehandler.close()

        print("Loaded in {}s.".format(time.time() - t))
    else:
        filehandler = open(load_name, 'rb')
        error = pickle.load(filehandler)
        filehandler.close()

    errors_sorted = np.empty((0, 2))
    for idx, e in enumerate(error):
        if len(e) > 0:
            index = np.full((len(e), 1), idx)
            #print(e.reshape((len(e), 1)))
            #print(index)
            new_e = np.concatenate((e.reshape(len(e), 1), index), axis=1)
            errors_sorted = np.vstack((errors_sorted, new_e))
    errors_sorted = np.flip(errors_sorted[np.argsort(errors_sorted[:, 0]), :], 0)
    print(errors_sorted[0:30, :])

    mu = 0
    variance = 1
    sigma = 7/3
    x = np.linspace(0, mu + 4 * sigma, 100)

    bins = 100
    print((np.amax(errors_sorted[:, 0]) - np.amin(errors_sorted[:, 0]))/bins)

    plt.hist(errors_sorted[:, 0], bins=50, density=True)
    plt.plot(x, mlab.normpdf(x, mu, sigma)*2)
    plt.show()




