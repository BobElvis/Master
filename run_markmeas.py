from PyQt5.QtWidgets import QApplication
from gui.appcreate import AppCreate, MarkData
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 1000

from dataset import *
from detection.detection import Detector, DetectData
import dataloader
from tracking import *
import sys
from measinit.boundary import create_boundary
from util import set_excepthook
from gui import background
from analyze import *
import analyze2



if __name__ == '__main__':
    set_excepthook()
    #sys.setrecursionlimit(sys.getrecursionlimit() * 100)

    # DATASET:
    folder = '2018-06-01 2018-06-07'
    dataset = DatasetFolder(ROOT_FOLDER, folder)
    dt = dataset.config.dt
    #i_start = 10792  # 10804
    i_start = 0
    i_end = 28800
    analyze = True

    # --- General parameters:
    detect_data = DetectData(dataset.config)
    bg = background.createBackground(detect_data, None, close_range=True)

    # --- Boundary:

    detector = Detector(dataset, detect_data, 34, 800, 1.0)
    detector.areaMinCnt = 34
    detector.enable_splitting = False
    detector.filter_multiples = False

    data_loader = dataloader.SimpleDataloader(detector)
    #meas_model = MeasurementModel(0.3, 10/3, 5, 0, 0, constant_acc=False)
    #meas_model = MeasurementModel(0.35, 8.5/3, 5, 0, 0, 5.63)
    #meas_model = MeasurementModel(0.6, 8.5/3, 5, 0, 0, 10)
    #meas_model = MeasurementModel(0.75, 8.5/3, 5, 0, 0, 5.63)
    meas_model = MeasurementModel(0.6, 8.5/3, 5, 0, 0, 5.63)
    #meas_model = MeasurementModel(0.75, 8/3, 5, 0, 0, 7.5)
    gate = TrackGate2(0.99)
    #gate = None

    mark_data = MarkData(dataset, data_loader, meas_model)
    mark_data.load()

    print("Last scan: {}".format(mark_data.get_last_scan_idx()))

    if analyze:
        split = 0
        merged = 0
        resolved = 0
        tot_meas = 0

        misdetects = 0
        n_tracks = 0
        n_scans = i_end - i_start

        data = extract_data(mark_data, i_start, i_end)
        #util.save_data(data, "temp_data/markmeasdata")
        #data = util.load_data("temp_data/markmeasdata")
        checked, tot_meas, clutter, assigned = data

        assignments = checked.values()
        for meas_indices, n_merge, first, last in assignments:
            if len(meas_indices) > 1 and n_merge > 1:
                print("WARNING: Merge and Split")
            elif len(meas_indices) > 1:
                split += len(meas_indices)
            elif n_merge > 1:
                merged += 1
            else:
                resolved += 1

        # # Extract:
        detections_resolved = [val for val in assignments if len(val[0]) == 1 and val[1] == 1]
        detections_split = [val for val in assignments if len(val[0]) > 1]
        detections_merge = [val for val in assignments if val[1] > 1]
        detections_first = [val for val in assignments if val[2]]
        detections_last = [val for val in assignments if val[3]]

        #---- Area filter:
        #clutter_area(mark_data, assigned, clutter)
        clutter_area2(mark_data, i_end, detections_first, clutter, assigned)

        ####### Multiple Filter:
        #analyze_multiple(mark_data, detect_data, i_end, clutter, assigned)

        # # Error splits:
        #detections_resolved_2 = [mark_data.get_detections(ind) for ind in extract(detections_split, key=lambda x: x[0])]
        #calcErrorSplit(detections_resolved_2)

        # Meas init/delete location:
        #arr = detections_last + detections_first
        #det_arr = mark_data.get_detections(flatten(arr, key=lambda x: x[0]))
        #plot_detections(det_arr, centroid=True, detect_data=detect_data)

        # Track data:
        #source_indices = [val[0][0] for val in detections_first]
        #tracks_d_idx = [mark_data.get_detection_track_with_idx(source_idx)[0] for source_idx in source_indices]
        #analyze2.analyze_statistics(mark_data, detect_data, tracks_d_idx, i_end, set())
        #print("----")
        print("****")

        #_, filter_tracks, removed_meas = analyze2.get_tracks(mark_data, detect_data, i_end, (34, 800, 1.0))
        #clutter_list, assign_list = analyze2.analyze_statistics(mark_data, detect_data, filter_tracks, i_end, removed_meas)
        #analyze2.running_density(mark_data, clutter_list, i_end, assign_list, filter_tracks)

        #analyze_plant_noise(source_indices, mark_data, meas_model)
        #analyze_track_prob([mark_data.get_detection_track(val[0][0])[0] for val in detections_first])

        # Location of clutter:
        #img_clutter = plot_detections(mark_data.get_detections(clutter), centroid=True, plot=True, detect_data=detect_data, bg=bg)
        # img_detect = plot_detections(mark_data.idx_to_detection(extract(assigned, key=lambda x:x[0])), centroid=False, plot=False)
        # img_sum = (img_detect + img_clutter)
        # alpha = img_sum == 0
        # img_sum[alpha] = 100000
        # img_ratio = img_clutter/img_sum
        # util.show_heatmap(img_ratio, bg, alpha)
        sys.exit(0)

    # --------- Start app:
    bg = background.createBackground(detect_data, None)
    app = QApplication(sys.argv)
    #ex = DynApp(dl, background, i_start, True)
    ex = AppCreate(data_loader, mark_data, bg, meas_model, gate, i=i_start)
    sys.exit(app.exec_())