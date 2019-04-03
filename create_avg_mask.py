from PyQt5.QtWidgets import QApplication
from gui.appcreate import AppCreate, MarkData
import dataset as dataset
import detection.detection as detection
import numpy as np
import util
import cv2
import gc
import time
import pickle
import matplotlib.pyplot as plt
import gui.background as background


if __name__ == '__main__':

    create = False  # Used to load a cached avg_img.
    save = False

    # SETTINGS:
    #folder = '2018-05-31'
    folder = '2018-06-01 2018-06-07'
    load_name = "temp_data/{}-avg_img_temp.obj".format(folder)
    dataset = dataset.DatasetFolder(dataset.ROOT_FOLDER, folder)
    detect_data = detection.DetectData(dataset.config)
    dataloader = detection.DetectionBase(dataset)
    bg = background.createBackground(detect_data, None)

    # SETTINGS:
    i_start = 0
    i_end = len(dataset)
    p = 0.01  # Should at least be > targets_per_scan for safety
    out_folder = "land"
    show_output = True
    fill = True

    # Go through samples:
    if create:
        t = time.time()
        print("i=({},{}). {} elements in dataset".format(i_start, i_end, len(dataset)))
        sum_img = np.zeros((detection.RAW_SIZE, detection.RAW_SIZE))
        sample_indices = range(i_start, i_end)
        n = len(sample_indices)
        for idx, data_idx in enumerate(sample_indices):
            if idx % (n/30) < 1:
                gc.collect()
                print("{}/{}".format(idx, n))
            radar_img = dataloader.load_radar(data_idx)
            sum_img += radar_img
        avg_img = sum_img / (n * np.amax(dataloader.load_radar(i_start)))

        # Saving the avg-img:
        filehandler = open(load_name, 'wb')
        pickle.dump(avg_img, filehandler)
        filehandler.close()

        print("Loaded in {}s.".format(time.time() - t))
    else:
        filehandler = open(load_name, 'rb')
        avg_img = pickle.load(filehandler)
        filehandler.close()

    avg_mask_uf = avg_img > p

    # Filling:
    avg_mask_img = avg_mask_uf.astype('uint8')
    _, cnts, _ = cv2.findContours(avg_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(avg_mask_img, cnts, -1, 255, cv2.FILLED)
    avg_mask_filled = avg_mask_img > 0

    if save:
        avg_mask = avg_mask_filled if fill else avg_mask_uf
        detection.save_avg_mask(avg_mask, dataset, p, i_start, i_end)

    # Visualize data:
    if show_output:
        bg = None
        plt.subplot(2, 2, 1)
        util.show_heatmap(avg_img, bg, show=False)
        plt.title("Frequency of detections in cells.")
        plt.subplot(2, 2, 2)
        util.show_heatmap(avg_img > p, bg, show=False)
        plt.title("Unfilled mask for p={:.2f}".format(p))
        plt.subplot(2, 2, 3)
        util.show_heatmap(avg_mask_filled, bg, show=False)
        plt.title("Filled mask for p={:.2f}".format(p))
        avg_img[avg_mask_filled] = p
        plt.subplot(2, 2, 4)
        util.show_heatmap(avg_img, bg, show=False)
        plt.title("Frequency ouside of filled mask, p={:.2f}".format(p))
        plt.tight_layout()
        plt.show()





