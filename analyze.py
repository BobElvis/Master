from detection.dataconfigs import RADAR_IMG_SIZE
import numpy as np
import cv2
import util
from detection.detection import DetectData
import matplotlib.pyplot as plt
from gui.appcreate import MarkData
from typing import Iterable, List
from model.model import *
import math
import detection.ghosts as multiple


def flatten(l, key):
    new_list = []
    key = (lambda x: x) if key is None else key
    for values in l:
        elements = key(values)
        for e in elements:
            new_list.append(e)
    return new_list


def extract(l, key):
    return [key(e) for e in l]


def extract_data(mark_data: MarkData, i_start, i_end):
    checked = dict()  # ((meas_idx,), n_merge, first, last)
    tot_meas = 0
    clutter = []
    assigned = set()

    for scan_i in range(i_start, i_end):
        tot_meas += len(mark_data.scan_meas_idx_map[scan_i])
        scan_data = mark_data.data[scan_i]

        for source, value in scan_data.items():
            if value is None:  # End of track.
                continue

            if len(value) == 0:
                continue

            idx_assignment = value[0]

            first = source in value
            last = mark_data.data[scan_i + 1][source] is None

            check_val = checked.get(idx_assignment)
            num = 1 if check_val is None else check_val[1] + 1
            checked[idx_assignment] = (value, num, first, last)

            # Make assignment
            assign_idx = value[0]
            if len(value) > 1:
                i_max = 0
                max_area = 0
                for i, det in enumerate(mark_data.get_detections(value)):
                    if det.area > max_area:
                        i_max = i
                assign_idx = value[i_max]
            assigned.add(assign_idx)

        # Find clutter:
        for meas_idx in mark_data.scan_meas_idx_map[scan_i]:
            if meas_idx not in assigned:
                clutter.append(meas_idx)

    return checked, tot_meas, clutter, list(assigned)


################################################################
###############################################################


def find_track_stats(tracks):
    data = []
    for track in tracks:
        # track: [[?, ...], ...]
        track_length = len(track)
        misdetects = 0
        miss_row = 0
        max_miss_row = 0
        for entry in track:
            # entry: [?,...]
            miss = len(entry) == 0
            miss_row = miss_row + 1 if miss else 0
            max_miss_row = max(miss_row, max_miss_row)
            misdetects += 1 if miss else 0
        data.append((track_length, misdetects, max_miss_row))
    data = np.array(data)

    # Extract:
    track_lengths = data[:, 0]
    max_miss_row = data[:, 2]
    P_O = data[:, 1] / track_lengths
    P_X = 1 / track_lengths
    P_D = 1 - P_O - P_X

    return P_D, P_O, P_X, track_lengths, max_miss_row


def get_tracks_removed(track_detect_ind, removed_set):
    new_tracks = []
    missed_tracks = []
    mis_detects_temp = []
    for track in track_detect_ind:
        # track: [[(detection, idx),...], ...]
        new_track = []
        mis_detects_temp.clear()

        # Create new track:
        for entry in track:
            # entry: [(detection, idx), ...)

            # Check if all entries are missed
            missed = True
            for e in entry:
                # e: (detection, idx)
                if e[1] not in removed_set:
                    missed = False
                    break

            # Determine if to append.
            if not missed:
                new_track.extend(mis_detects_temp)
                new_track.append(entry)
                mis_detects_temp.clear()
            elif len(new_track) > 0:
                mis_detects_temp.append(entry)

        # if len(new_track) < len(track):
        #    print("Diff: {}".format(len(track) - len(new_track)))

        # Handle if new source is split. Choose max area one.
        if len(new_track) > 1:
            first_entry = new_track[0]
            if len(first_entry) > 1:
                entry_max = max(first_entry, key=lambda x: x[0].area)
                new_track[0] = [entry_max]
            new_tracks.append(new_track)
        else:
            print("MISSED")
            missed_tracks.append(track)
    return new_tracks, missed_tracks


def analyze_multiple(mark_data: MarkData, detect_data: DetectData, i_end, clutter_indices, assignment_indices):
    area_values = np.arange(700, 1020, 20)
    # range_values = np.array([1.0, 1.05, 1.1])
    range_values = np.array([1.0, 1.1, 1.2, 1.5, 2.0])
    poss = []
    is_clutter = []

    n_area = len(area_values)
    n_range = len(range_values)

    data = np.zeros((i_end, n_area, n_range, 2))
    tot = np.zeros((i_end, 2))
    n_meas = np.full(i_end, fill_value=False)

    for scan_i in range(i_end):
        print(scan_i)
        target_ind = mark_data.scan_meas_idx_map[scan_i]
        n = len(target_ind)
        n_meas[scan_i] = n >= 2
        if n < 2:
            continue

        # Get the data:
        poss.clear()
        is_clutter.clear()
        cnts = []
        areas = []
        for target_i in target_ind:
            detection = mark_data.reverse_map.get(target_i)[2]
            poss.append(detection.pos)
            cnts.append(detection.cnt)
            areas.append(detection.area)
            is_clutter.append(target_i in clutter_indices)
        pos_arr = np.array(poss)
        clutter_arr = np.array(is_clutter)
        detect_arr = np.logical_not(clutter_arr)
        data_m, dists = multiple._get_multiple_data(detect_data, pos_arr, cnts, areas)

        # Clutter:
        clutter = np.sum(clutter_arr)
        tot[scan_i, :] = (n - clutter, clutter)

        # Run multiple
        for i, area_val in enumerate(area_values):
            for j, range_val in enumerate(range_values):
                multiples = multiple._mark_multiple(dists, data_m, area_val, range_val)
                keep = np.logical_not(multiples)
                n_clutter = np.sum(np.logical_and(clutter_arr, keep))
                n_detect = np.sum(np.logical_and(detect_arr, keep))
                data[scan_i, i, j, :] = (n_detect, n_clutter)

    # Only n >= 2:
    # data = data[n_meas, :, :, :]
    # tot = tot[n_meas, :]

    data_sum = data.sum(axis=0)
    tot = np.sum(tot, axis=0)
    TP = data_sum[:, :, 0]
    FN = tot[0] - data_sum[:, :, 0]
    FP = data_sum[:, :, 1]
    TN = tot[1] - data_sum[:, :, 1]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    # plt.plot(area_values, data_sum.reshape((n_area, 2*n_range)))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    lines = plt.plot(1 - specificity, sensitivity)
    for line, range_val in zip(lines, range_values):
        line.set_label("r = {:.2f}".format(range_val))

    # TEXT VALUES:
    text_vals = [100, 500, 800, 900, 1000]
    for i in range(0, len(area_values)):
        if area_values[i] not in text_vals:
            continue
        for j in range(len(range_values)):
            ax.text(1 - specificity[i, j], sensitivity[i, j],
                    "{:.0f}".format(area_values[i]))
    plt.grid()
    plt.legend()
    plt.show()


def clutter_area2(mark_data: MarkData, i_end, detections_first, clutter, assignment_ind):
    source_indices = [val[0][0] for val in detections_first]
    tracks = [mark_data.get_detection_track_with_idx(val[0][0])[0] for val in detections_first]

    # Create data from clutter:
    clutter_detections = mark_data.get_detections(clutter)
    data_clutter = np.array([(d.pos[0], d.pos[1], d.area) for d in clutter_detections])

    print(data_clutter.shape)

    # Create data from assignment (indices)
    data_assigned = []
    for assign_idx in assignment_ind:
        detection = mark_data.get_detection(assign_idx)
        data_assigned.append((detection.pos[0], detection.pos[1], detection.area, assign_idx))
    data_assigned = np.array(data_assigned)

    # Get min, max:
    min_area = min(np.amin(data_clutter[:, 2]), np.amin(data_assigned[:, 2]))
    max_area = max(np.amax(data_clutter[:, 2]), np.amax(data_assigned[:, 2]))

    # Create threshold values:
    values = np.arange(20, 81, 1)

    # data: [thresh, detect/clutter ratio, PD, TrackLen, max_miss_row]
    data_thresh = np.empty((len(values), 8))

    # Create data:
    for i, thresh in enumerate(values):
        FP = np.sum(data_clutter[:, 2] >= thresh)
        is_detected = data_assigned[:, 2] >= thresh
        TP = np.sum(is_detected)

        # Create set of removed indices
        removed_ind = data_assigned[np.logical_not(is_detected), :][:, 3]
        assert len(removed_ind) == data_assigned.shape[0] - TP
        removed = set(removed_ind)

        # Analyze track:
        tracks_removed, missed_tracks = get_tracks_removed(tracks, removed)

        if len(missed_tracks) > 0:
            print("Missed tracks: {} for {}".format(len(missed_tracks), thresh))
            # print(missed_tracks[0][0][0][1])
            source_missed = missed_tracks[0][0][0][1]
            print(" - Missed: {}".format(mark_data.reverse_map[source_missed]))

        PD, PO, PX, track_len, miss_row = find_track_stats(tracks_removed)

        PDm = np.mean(PD)
        PDw = np.amin(PD)
        # POm = np.mean(PO)
        # PXm = np.mean(PX)

        track_len_mean = np.mean(track_len)
        track_len_w = np.amin(track_len)
        max_miss = np.amax(miss_row)
        data_thresh[i, :] = (thresh, TP, FP, PDm, PDw, track_len_mean, track_len_w, len(missed_tracks))

    # Plot 1:
    TP = data_thresh[:, 1]
    FN = len(assignment_ind) - TP
    FP = data_thresh[:, 2]
    TN = len(clutter) - FP

    sens = TP / (TP + FN)
    spec = TN / (TN + FP)

    #plt.plot(values, FP)
    #plt.plot(values, TN)
    plt.plot(values, spec)
    plt.plot(values, sens)
    plt.show()

    if True:
        ax = plt.subplot(1, 1, 1)
        plt.plot(1 - spec, sens)
        text_vals = [20, 30, 40, 50, 60, 70, 80]
        for i in range(0, len(values)):
            if values[i] not in text_vals:
                continue
            ax.text(1 - spec[i], sens[i],
                    "{:.0f}".format(values[i]), bbox=dict(facecolor='white', alpha=0.9))
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        plt.show()

    # Plot:
    if False:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('A min [pixels^2]')
        ax1.set_ylabel('PD')
        # ax1.plot(1 - roc[:, 1], roc[:, 1], color=color)
        l = ax1.plot(data_thresh[:, 0], data_thresh[:, 4], color='tab:orange')[0]
        l.set_label("Lowest Track PD")
        # ax1.plot(data_thresh[:, 0], data_thresh[:, ])
        ax1.tick_params(axis='y')
        ax1.set_xticks(np.arange(values[0], values[-1] + 5, 20))
        ax1.grid()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:cyan'
        ax2.set_ylabel('Track Length, Number of Missed Tracks')  # we already handled the x-label with ax1
        l = ax2.plot(data_thresh[:, 0], data_thresh[:, 6], color=color)[0]
        l.set_label("Shortest Track")
        l = ax2.plot(data_thresh[:, 0], data_thresh[:, 7], color='tab:blue')[0]
        l.set_label("Missed tracks")
        ax2.tick_params(axis='y')

        ax1.legend(loc='center left')
        ax2.legend(loc='right')

        # TEXT VALUES:
        # for i in range(len(values)):
        #    ax1.text(1 - roc[i, 1], roc[i, 1], "{}".format(thresh))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


def plot_detections(detections, centroid=False, bg=None, normalize=False, detect_data: DetectData = None, plot=True):
    img = np.full((RADAR_IMG_SIZE, RADAR_IMG_SIZE), fill_value=0)
    alpha = None
    if centroid:
        centroids = np.array([d.pos for d in detections])
        centroid_scaled = detect_data.scale_measurements(centroids, reverse=True)
        img, _, _ = np.histogram2d(centroid_scaled[:, 0], centroid_scaled[:, 1], bins=128,
                                   range=((0, RADAR_IMG_SIZE), (0, RADAR_IMG_SIZE)), normed=True)
        img = img.T
        alpha = cv2.resize(detect_data.full_mask.astype(np.uint8), img.shape) > 0
    else:
        img_temp = img.copy()
        for detection in detections:
            cv2.drawContours(img_temp, (detection.cnt,), 0, 1, cv2.FILLED)
            np.add(img, img_temp, out=img)
    if plot:
        util.show_heatmap(img, bg, axes=False, alpha_mask=alpha)
    return img


def analyze_plant_noise(source_indices: List[int], mark_data: MarkData, meas_model):
    acc = True
    data = []
    data2 = []
    data3 = []
    speed = []
    speed_init = []
    scan_ind = []
    base3 = np.array((0.5 * (meas_model.dt ** 2), meas_model.dt))

    tracks = [(mark_data.get_track_nodes(source_idx), source_idx) for source_idx in source_indices]
    for track, source_idx in tracks:
        covariance = []
        change = []
        point_acc = []
        track_scan_idx = []
        for i in range(1, len(track) - 1):
            if track[i - 1].measurement is not None and track[i].measurement is not None:
                if track[i].scan_idx == 13162:
                    print("TEST")
                    print(track[i-1].measurement.pos)
                    print(track[i].measurement.pos)
                    print(track[i-1].measurement.pos - track[i].measurement.pos)
                    print(np.linalg.norm(track[i-1].measurement.pos - track[i].measurement.pos))
                    print(meas_model.dt)

                s1 = np.linalg.norm((track[i-1].measurement.pos - track[i].measurement.pos))/meas_model.dt
                s2 = np.linalg.norm((track[i].est_posterior[1], track[i].est_posterior[3]))
                s = s1
                speed.append((s, i))
                if i == 1:
                    speed_init.append((s, track[i].scan_idx))
            if track[i + 1].measurement is None or track[i].measurement is None or track[i - 1].measurement is None:
                continue
            diff = track[i + 1].est_posterior - track[i].est_prior
            w_k = diff.reshape((4, 1))
            Q_k = w_k.dot(w_k.T)

            diff = track[i + 1].measurement.pos - 2 * track[i].measurement.pos + track[i - 1].measurement.pos
            diff = diff / (meas_model.dt ** 2)

            point_acc.append(diff)
            covariance.append(Q_k[0:2, 0:2])
            covariance.append(Q_k[2:4, 2:4])
            change.append(w_k[0:2])
            change.append(w_k[2:4])
            track_scan_idx.append(track[i].scan_idx)
            track_scan_idx.append(track[i].scan_idx)
            track_scan_idx.append(track[i].scan_idx)
            track_scan_idx.append(track[i].scan_idx)

        scan_ind.extend(track_scan_idx)
        data.extend(covariance)
        data2.extend(change)
        data3.extend(point_acc)
    data = np.array(data)
    data2 = np.array(data2)
    data3 = np.array(data3)
    scan_ind = np.array(scan_ind)
    speed = np.array(speed)
    speed_init = np.array(speed_init)
    print("ALL:")
    # plt.hist(speed, bins=500, normed=True)
    # plt.show()

    if acc:
        diff = np.mean((data2 / base3).reshape((-1, 2)), axis=1)
        acc_sortarg = np.argsort(diff)
        print(np.flip(diff[acc_sortarg], axis=0)[:20])
        print(np.flip(scan_ind[acc_sortarg], axis=0)[:20])

        # diff = (data2/base3).reshape((-1, 2))
        plt.hist([diff], bins=200, normed=True, range=(-3, 3))
        l = plot_normal(0, 0.6)
        l.set_label("Normal(0, 0.6)")
        plt.xlabel("Acceleration")
        plt.ylim(0, 0.25)
        plt.legend()
        plt.show()
    else:
        np.set_printoptions(suppress=True)
        speed_sort = np.flip(speed_init[np.argsort(speed_init[:, 0]), :],axis=0)[:10, :]
        print(speed_sort)
        plt.hist([speed[:, 0]], normed=True, bins=30)
        plt.xlabel("Speed [m/s]")
        plt.ylabel("Frequency")
        plt.show()
    return


def analyze_track_prob(tracks):
    data = []
    for track in tracks:
        track_length = len(track)  # Not using the first measurement
        misdetects = 0
        miss_row = 0
        max_miss_row = 0
        for entry in track:
            miss = len(entry) == 0
            miss_row = miss_row + 1 if miss else 0
            max_miss_row = max(miss_row, max_miss_row)
            misdetects += 1 if miss else 0
        data.append((track_length, misdetects, max_miss_row))
    data = np.array(data)
    P_O = data[:, 1] / data[:, 0]
    P_X = 1 / (data[:, 0])
    P_D = 1 - P_O - P_X

    P_Dm = np.mean(P_D)
    P_Om = np.mean(P_O)
    P_Xm = np.mean(P_X)

    print("P = {}, {}, {}. S: {}".format(P_Dm*100, P_Om*100, P_Xm*100, P_Dm + P_Om + P_Xm))

    s = 6
    # y = (1-P_Dm)/(P_Xm*(s+1))
    y = 1 / (P_Xm * (2 * s + 1) + P_Dm)
    P_Xn = P_Xm * y
    P_On = P_Xn * s
    P_Dn = 1 - P_Xn - P_On
    print("P = {}, {}, {}".format(P_Dn, P_On, P_Xn))
    print("Max: PO:{}".format(np.amax(P_O)))
    print("Median: {}, {}, {}".format(np.median(P_D), np.median(P_O), np.median(P_X)))
    print("Avg track length: {}".format(np.mean(data[:, 0])))

    tot_plot = 4
    plt.subplot(1, tot_plot, 1)
    plt.hist(P_D, bins=30)
    plt.title("PD")

    plt.subplot(1, tot_plot, 2)
    plt.hist(P_X, bins=30)
    plt.title("PX")

    plt.subplot(1, tot_plot, 3)
    plt.hist(P_O, bins=30)
    plt.title("PO")

    plt.subplot(1, tot_plot, 4)
    plt.hist(data[:, 2], bins=np.amax(data[:, 2]))
    plt.title("Max consecutive misd.")
    plt.show()


def ROCarea():
    pass


def plot_normal(mu, sigma, half=False, linewidth=2.0):
    start = 0 if half else mu - 4 * sigma
    x = np.linspace(start, mu + 4 * sigma, 100)
    f = 2 if half else 1
    return plt.plot(x, plt.mlab.normpdf(x, mu, sigma) * f, linewidth=linewidth)[0]


def calcErrorSplit(assignments):
    from detection.geometry import centroid_of_centroids
    data = []
    for detections in assignments:
        pos, area = centroid_of_centroids(detections)
        data_i = np.array([(pos[0], pos[1], d.pos[0], d.pos[1], d.area) for d in detections])
        min_error = np.argmin(np.linalg.norm([data_i[:, 0:2] - data_i[:, 2:4]], axis=1))
        data.append(data_i[min_error, :].reshape(1, -1))

    data = np.concatenate(data, axis=0)
    errors = np.linalg.norm(data[:, 0:2] - data[:, 2:4], axis=1)
    errors = np.abs((data[:, 0:2] - data[:, 2:4]).reshape((-1)))
    area = np.sqrt(data[:, 4] / np.pi)
    bins = 30
    plt.subplot(1, 1, 1)
    plt.hist(errors, bins=bins, normed=True, range=(0, 12))
    plot_normal(0, 8.5 / 3, linewidth=4.0, half=True)
    # plt.subplot(1, 3, 2)
    # plt.hist(area, bins=bins)
    # plt.subplot(1, 3, 3)
    # plt.hist(errors/area, bins=bins)
    plt.show()
