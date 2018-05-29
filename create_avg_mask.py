import detection.dataset as dataset
import detection.detection as detection
import numpy as np
import util


if __name__ == '__main__':
    # SETTINGS:
    i_start = 0
    i_end = 200
    rotation = detection.ROTATION
    p = 0.2  # Should at least be > targets_per_scan for safety
    day = '2018-05-28'
    partition = 0
    out_folder = "land"

    dataset = dataset.Dataset(dataset.ROOT_FOLDER, day, partition)
    dataloader = detection.DetectionBase(dataset, rotation)

    print("{} elements in dataset".format(len(dataset)))

    # Go through samples:
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
    avg_mask = avg_img > p
    detection.save_avg_mask(avg_mask, rotation, p, day, partition, i_start, i_end)

    # Visualize data:
    util.show_heatmap(avg_img)
    avg_img[avg_img > 0.2] = 0.2
    util.show_heatmap(avg_img)
    util.show_heatmap(avg_mask)





