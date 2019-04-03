from PyQt5.QtWidgets import QApplication
from gui.appcreate import AppCreate, MarkData
from gui.appmht import TrackApp

from dataset import *
from detection.detection import Detector, DetectData
import dataloader
from tracking import *
from mht.mhtdata import MhtData
import sys
from measinit.meas_init import MeasInitBase
from measinit.boundary import create_boundary
from util import set_excepthook, show_heatmap
import mht.pruning as pruning
from mht.mht import MHT
import timer
import threading
from gui import background
from detection.ghosts import mark_multiple
from dataset import *
from util import set_excepthook
import analyze
import analyze2
import matplotlib.pyplot as plt
import itertools
import util


TEMP = []


def is_track_valid(track: List[TrackNode]):
    poss = np.empty((len(track), 2))
    for i, track_step in enumerate(track):
        x, y = track_step.est_posterior[0], track_step.est_posterior[2]
        poss[i, :] = x, y
        #print("x={:.2f}, y={:.2f}".format(x, y))
    centroid = np.mean(poss, axis=0)
    diff = np.linalg.norm(poss - centroid, axis=1)
    return np.amax(diff) >= 5


def examine_temp():
    np.set_printoptions(suppress=True)
    arr = np.array(TEMP)
    arr = arr[arr[:, 1] >= 5, :]
    arr = arr[np.argsort(arr[:, 1]), :][0:20]
    print(arr)


def to_scan_lists(tracks: List[List[TrackNode]], i_end):
    scan_list = [[] for _ in range(i_end)]
    for track in tracks:
        for node in track:
            if node.scan_idx >= i_end:
                break
            scan_list[node.scan_idx].append(node)
    return scan_list


def ospa_dist(n1, n2):
    return np.linalg.norm(n1.est_posterior - n2.est_posterior, axis=0)


def ospa(nodes_m, nodes_n, ospa_c, ospa_p):
    m = len(nodes_m)
    n = len(nodes_n)
    permutations_n = itertools.permutations(nodes_n, m)
    min_score = 1e100
    tracks_min = 0
    for permutation in permutations_n:
        score = 0
        tracks = 0
        for i in range(m):
            d = ospa_dist(nodes_m[i], permutation[i])
            tracks += 1 if d < ospa_c else 0
            score += min(d, ospa_c)**ospa_p

        if score < min_score:
            min_score = score
            tracks_min = tracks
    min_score += (ospa_c ** ospa_p)*(n-m)
    return (min_score/n)**(1/ospa_p), tracks_min


def main():
    # DATASET:
    folder = '2018-06-01 2018-06-07'
    dataset = DatasetFolder(ROOT_FOLDER, folder)
    dt = dataset.config.dt
    detect_data = DetectData(dataset.config)
    i_end = 28800
    detect_parameters = (34, 800, 1.0)  # areamin, areamult, rangemult
    ospa_c = 7.5
    ospa_p = 2
    rmse_p = 2
    experiment = 21
    OSPA_ref = 0.6733562422266803  # Experiment 0

    #mht_data = MhtData(dataset, settings=None)
    #mht_data.restore()

    ############ MARK DATA #############:
    mark_meas_model = MeasurementModel(0.6, 0.2833, dt, 0, 0, 5.63)
    mark_data = MarkData(dataset, None, mark_meas_model)
    mark_data.load()
    tracks_truth = analyze2.get_tracks(mark_data, detect_data, i_end, detect_parameters)[0]
    scan_truth = to_scan_lists(tracks_truth, i_end)

    ############ MHT ##################:
    mht_meas_model = MeasurementModel(0.6, 0.2833, dt, 0, 0, 5.63)
    mht_data = MhtData(dataset, None)
    sett = mht_data.restore(experiment)
    print("----- SETTINGS -------")
    print("".join(sett))
    print("----- END SETTINGS ---")
    print("Experiment {}".format(experiment))

    # 1. Load Ground Truth
    # 2. Apply Filters and get new tracks
    # 3. Save as it can be loaded next time.
    # 4. Save it for each timestep.

    # Get Tracks:
    # 1. Run MHT
    # 2. Examine for still tracks.
    # 3. Find a way to remove them.
    # 4. Extract tracks, and save a node for each timestep.

    clusters = mht_data.get_clusters(i_end) + mht_data.get_clusters_dead(i_end)
    tracks_mht = []
    worst_K = []
    for c in clusters:
        best_hyp = c.leaves[0]
        worst_K.append((best_hyp.K_max, best_hyp.ratio_max))
        for track_node in best_hyp.track_nodes + best_hyp.track_nodes_del:
            track = track_node.getTrack(0, i_end)
            smooth_track = mht_meas_model.RTS(track)
            if len(track) > 0 and is_track_valid(smooth_track):
                tracks_mht.append(track)
    scan_mht = to_scan_lists(tracks_mht, i_end)

    worst_K = np.array(worst_K)
    print("Worst K")
    print(np.flip(worst_K[np.argsort(worst_K[:, 0]), :], axis=0)[:10])
    print("Worst r")
    print(np.flip(worst_K[np.argsort(worst_K[:, 1]), :], axis=0)[:10])
    print("Number of tracks: MHT: {} | Truth: {}".format(len(tracks_mht), len(tracks_truth)))

    # COMPARE:
    number_tracks = np.zeros((i_end, 3))
    ospa_scores = np.zeros((i_end))
    for i, (nodes_truth, nodes_mht) in enumerate(zip(scan_truth, scan_mht)):
        m = len(nodes_truth)
        n = len(nodes_mht)
        if m == 0 and n == 0:
            ospa_score, ospa_tracks = 0, 0
        elif m <= n:
            ospa_score, ospa_tracks = ospa(nodes_truth, nodes_mht, ospa_c, ospa_p)
        else:
            ospa_score, ospa_tracks = ospa(nodes_mht, nodes_truth, ospa_c, ospa_p)
        ospa_scores[i] = ospa_score

        number_tracks[i, :] = len(nodes_truth), len(nodes_mht), ospa_tracks

    # RMSE:
    diff = number_tracks[:, 0] - number_tracks[:, 1]
    score = np.power(np.mean(np.power(diff, rmse_p)), 1/rmse_p)
    print("RMSE: {:.2f}, OSPA: {:.2f}".format(score*1000, np.mean(ospa_scores)*1000))

    # Plot:
    indices = np.arange(0, i_end)

    #plt.plot(indices, ospa_scores)
    #plt.plot(indices, diff)
    #plt.show()

    # Plot numbers:
    #plt.plot(np.arange(0, i_end), number_tracks[:, 0] - number_tracks[:, 1])
    #plt.show()



if __name__ == '__main__':
    set_excepthook()
    main()
    #sys.setrecursionlimit(sys.getrecursionlimit() * 100)


