import detection.dataset as dataset
import detection.detection as detection
import numpy as np
import util
import cv2


if __name__ == '__main__':

    # SETTINGS:
    day = '2018-05-28'
    partition = 0

    dataset = dataset.Dataset(dataset.ROOT_FOLDER, day, partition)
    dataloader = detection.DetectionBase(dataset)

    # SETTINGS:
    i_start = 0
    i_end = len(dataset)
    p = 0.1  # Should at least be > targets_per_scan for safety
    out_folder = "land"
    show_output = True
    save = True
    fill = True

    # Go through samples:
    print("i=({},{})/{} elements in dataset".format(i_start, i_end, len(dataset)))
    sum_img = np.zeros((detection.RAW_SIZE, detection.RAW_SIZE))
    sample_indices = range(i_start, i_end)
    n = len(sample_indices)
    for idx, data_idx in enumerate(sample_indices):
        if idx % (n/10) < 1:
            print("{}/{}".format(idx, n))
        radar_img = dataloader.load_radar(data_idx)
        sum_img += radar_img

    # Threshold and save:
    avg_img = sum_img/(n*np.amax(dataloader.load_radar(i_start)))
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
        util.show_heatmap(avg_img)
        util.show_heatmap(avg_img > p)
        util.show_heatmap(avg_img > p/2)
        util.show_heatmap(avg_mask_filled)

        avg_img[avg_img > p] = p
        util.show_heatmap(avg_img)





