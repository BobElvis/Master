from detection.detection import DetectData
import matplotlib.pyplot as plt
from gui.appcreate import MarkData
from detection.ghosts import mark_multiple
import analyze
import numpy as np


def time_print(x):
    m, s = divmod(x, 60)
    return "{:.0f}m {:.0f}s".format(m, s)


def print_mean_min_max(name, data, precision, out_func=None):
    if out_func is None:
        out_func = lambda x: x
    p = precision
    print(("{} | avg: {"+p+"}, min: {"+p+"}, max: {"+p+"}").
          format(name, out_func(np.mean(data)), out_func(np.amin(data)), out_func(np.amax(data))))


def analyze_statistics(mark_data: MarkData, detect_data: DetectData, tracks_d_idx, i_end, removed):

    # Find the complete set of assignments (largest of split):
    n_split = 0
    merged = dict()  # (n_merge)
    assigned_all = set()
    assigned_split = set()
    for track in tracks_d_idx:
        for track_step in track:
            if len(track_step) == 0:
                continue
            if len(track_step) > 1:
                n_split += len(track_step)
                d_idx = max(track_step, key=lambda x: x[0].area)
            else:
                d_idx = track_step[0]
            assigned_split.add(d_idx[1])

            # All measurements:
            for entry in track_step:
                assigned_all.add(entry[1])

            # Merged:
            if len(track_step) == 1:
                idx_assignment = track_step[0][1]
                prev_val = merged.get(idx_assignment)
                num = 1 if prev_val is None else prev_val + 1
                merged[idx_assignment] = num

    # Check merged:
    n_merged = 0
    n_resolved = 0
    for _, num in merged.items():
        if num > 1:
            n_merged += 1
        else:
            n_resolved += 1

    # Find clutter:
    clutter_non_mht = []
    clutter_list = []
    for i in range(i_end):
        for meas_idx in mark_data.scan_meas_idx_map[i]:
            if meas_idx not in assigned_all and meas_idx not in removed:
                clutter_non_mht.append(meas_idx)
            if meas_idx not in assigned_split and meas_idx not in removed:
                clutter_list.append(meas_idx)

    tot_meas = n_split + n_merged + n_resolved + len(clutter_non_mht)

    # Track stats (Ignores the effect of merged measurements):
    P_D, P_O, P_X, track_lengths, max_miss_row = analyze.find_track_stats(tracks_d_idx)

    # Print info:
    print("Tracks | {}".format(len(tracks_d_idx)))
    print_mean_min_max("PD", P_D, ":.4f")
    print_mean_min_max("PO", P_O, ":.4f")
    print_mean_min_max("PX", P_X, ":.4f")
    print_mean_min_max("Track Length [scan]", track_lengths, ":.1f")
    print_mean_min_max("Track Length [time]", track_lengths*detect_data.data_config.dt, "", out_func=time_print)
    print("Split: {} | {}".format(n_split, n_split/tot_meas*100))
    print("Merged: {} | {}".format(n_merged, n_merged/tot_meas*100))
    print("Resolved: {} | {}".format(n_resolved, n_resolved/tot_meas*100))
    print("Clutter {} | {}".format(len(clutter_non_mht), len(clutter_non_mht)/tot_meas*100))
    print("Clutter MHT: {} | {}".format(len(clutter_list), len(clutter_list)/tot_meas*100))
    print("TOT: {}".format(n_split + n_merged + n_resolved + len(clutter_non_mht)))
    print("Clutter per scan: {:.5f} | {:.3g}".format(len(clutter_list)/i_end, len(clutter_list)/(i_end*detect_data.area())))
    print("New Targets per scan: {:.5f} | {:.3g}".format(len(tracks_d_idx)/i_end, len(tracks_d_idx)/(i_end*detect_data.area())))

    #Plot PD, PO, PX
    plot = True
    if plot:
        tot_plot = 3
        plt.subplot(1, tot_plot, 1)
        plt.hist(P_D, bins=30, color='tab:green')
        plt.title("PD")

        plt.subplot(1, tot_plot, 2)
        plt.hist(P_O, bins=30, color='tab:red')
        plt.title("PO")

        plt.subplot(1, tot_plot, 3)
        plt.hist(P_X, bins=30, color='tab:orange')
        plt.title("PX")
        plt.show()

    return clutter_list, assigned_split


def running_density(mark_data: MarkData, clutter, i_end, assigned, tracks):
    # Tracks:
    new_target_scan = np.zeros(i_end)
    for track in tracks:
        meas_idx = track[0][0][1]
        new_target_scan[mark_data.reverse_map[meas_idx][0]] += 1

    # Target scan
    target_scan = np.zeros(i_end)
    for meas_idx in assigned:
        target_scan[mark_data.reverse_map[meas_idx][0]] += 1

    # Clutter scan
    clutter_scan = np.zeros(i_end)
    for meas_idx in clutter:
        clutter_scan[mark_data.reverse_map[meas_idx][0]] += 1

    N_run = int(round(60*2/5))
    print(N_run)
    run_mean_new_target = np.convolve(new_target_scan, np.ones(N_run)/N_run, mode='valid')
    run_mean_target = np.convolve(target_scan, np.ones(N_run)/N_run, mode='valid')
    run_mean_clutter = np.convolve(clutter_scan, np.ones(N_run)/N_run, mode='valid')
    indices = np.arange(0, i_end+1-N_run)

    std_clutter = np.std(run_mean_clutter)
    std_new = np.std(run_mean_new_target)
    std_up_clutter = std_clutter + np.mean(run_mean_clutter)
    std_up_new = std_new + np.mean(run_mean_new_target)
    std_dw_clutter = - std_clutter + np.mean(run_mean_clutter)
    std_dw_new = - std_new + np.mean(run_mean_new_target)

    print("Mean + STD: {:.3f}, {:.3f}".format(std_up_clutter, std_dw_new))

    # Plot:
    fig, ax1 = plt.subplots()
    color = 'tab:orange'
    ax1.set_xlabel('Scan idx')
    ax1.set_ylabel(' ', color=color)
    # ax1.plot(1 - roc[:, 1], roc[:, 1], color=color)
    l = ax1.plot(indices, run_mean_clutter, color=color)[0]
    l.set_label("Clutter")
    l = ax1.plot(indices, np.ones(len(indices)) * std_up_clutter, color='tab:blue')[0]
    l.set_label("Mean + Std")
    l = ax1.plot(indices, np.ones(len(indices)) * std_dw_clutter, color='tab:blue')[0]
    l.set_label("Mean - Std")
    # ax1.plot(data_thresh[:, 0], data_thresh[:, ])
    ax1.set_ylim(bottom=-1, top=1.2)
    ax1.tick_params(axis='y', colors=color)
    #ax1.set_xticks(np.arange(run_mean_clutter[0], run_mean_clutter[-1] + 5, 20))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:cyan'
    ax2.set_ylabel(' ', color=color)  # we already handled the x-label with ax1
    l = ax2.plot(indices, run_mean_new_target, color=color)[0]
    l.set_label("New Targets")
    l = ax2.plot(indices, np.ones(len(indices)) * std_up_new, color='tab:red')[0]
    l.set_label("Mean + Std")
    l = ax2.plot(indices, np.ones(len(indices)) * std_dw_new, color='tab:red')[0]
    l.set_label("Mean - Std")
    ax2.set_ylim(bottom=0, top=0.5)
    ax2.tick_params(axis='y', colors=color)

    ax1.legend()
    ax2.legend()

    # TEXT VALUES:
    # for i in range(len(values)):
    #    ax1.text(1 - roc[i, 1], roc[i, 1], "{}".format(thresh))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    return
    l1 = plt.plot(indices, run_mean_clutter)[0]
    l2 = plt.plot(indices, run_mean_new_target)[0]
    l1.set_label("Clutter per scan")
    l2.set_label("New target per scan")
    plt.legend()
    plt.show()


def get_tracks(mark_data: MarkData, detect_data: DetectData, i_end, detection_parameters):
    # ----------- Extract sources:
    source_indices = []
    for scan_i in range(i_end):
        scan_data = mark_data.data[scan_i]
        for source_idx, meas_indices in scan_data.items():
            if meas_indices is None or source_idx not in meas_indices:
                continue
            source_indices.append(source_idx)

    # ---------- Get original tracks:
    old_tracks_detection = [mark_data.get_detection_track_with_idx(source_idx)[0] for source_idx in source_indices]

    # ---------- Find removed measurements:
    min_area, min_area_multiple, ratio_multiple = detection_parameters
    removed_tot = set()
    for scan_i in range(i_end):
        meas_indices = mark_data.scan_meas_idx_map[scan_i]

        poss = []
        cnts = []
        areas = []
        for target_i in meas_indices:
            dct = mark_data.reverse_map[target_i][2]
            poss.append(dct.pos)
            cnts.append(dct.cnt)
            areas.append(dct.area)
        poss = np.array(poss)

        is_multiple = mark_multiple(detect_data, poss, cnts, areas, min_area_multiple, ratio_multiple)

        for idx, target_i in enumerate(meas_indices):
            if (is_multiple is not None and is_multiple[idx]) or areas[idx] < min_area:
                removed_tot.add(target_i)

    # --------- Get tracks removed:
    new_tracks_detection = analyze.get_tracks_removed(old_tracks_detection, removed_tot)[0]

    # --------- Removed the index from the lists:
    new_tracks = [[[d for d, idx in entry] for entry in track] for track in new_tracks_detection]

    # [track_element0][entry0][target_idx] > scan_idx
    first_scan_ind = [mark_data.reverse_map[v[0][0][1]][0] for v in new_tracks_detection]
    tracks_nodes = [mark_data.create_track(d, first_scan_ind) for d, first_scan_ind in zip(new_tracks, first_scan_ind)]

    return tracks_nodes, new_tracks_detection, removed_tot