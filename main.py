from PyQt5.QtWidgets import QApplication
from gui.appmht import TrackApp, DynApp

from dataset import *
from detection.detection import Detector, DetectData
import dataloader
from tracking import *
from mht.mhtdata import MhtData, Settings
import sys
from measinit.meas_init import MeasInitBase
from util import set_excepthook, show_heatmap
import mht.pruning as pruning
from mht.mht import MHT
import timer
import threading
from gui import background


def main():
    set_excepthook()
    #sys.setrecursionlimit(sys.getrecursionlimit() * 100)

    # PARAMETERS:
    s = Settings()
    s.q_std = 0.4
    s.r_std = 8.5/3  # 99.8 % lies within +-8.5 meters.
    s.v_max = 5.76
    s.PG = 0.99

    # TUNE (targets, clutter, PD, PX):
    t = (0.01014, 0.0891*1.25, 0.9*0.9, 0.0546) # Avg
    #t = (0.01014, 0.0891, 0.9*0.9, 0.0546) # Avg
    t = (0.01014, 0.0891*1.25, 0.9*0.9, 0.0546) # Avg
    #t = (0.01514, 0.0891, 0.91, 0.0540) # Avg (pre filter)
    #t = (0.01014*0.75, 0.0891, 0.91, 0.0546) # Lowered targets per scan
    #t = (0.01014, 0.0891*1.5, 0.7, 0.025) # Lowered PD, PX
    #t = (0.01014, 0.0891*1.5, 0.9, 0.0546)  # Increased clutter
    #t = (0.01014, 0.0891*1.5, 0.7, 0.05)  # Increased clutter, low PD
    #t = (0.01014, 0.0891*2, 0.85, 0.05)  # Increased clutter

    # SET:
    s.targets_per_scan = t[0] #0.0005
    s.clutter_per_scan = t[1] #0.2  # 0.05
    s.P_D = t[2]  #0.70
    s.P_X = t[3]  #0.025

    # s.area_min = 34
    # s.area_multiple = 800
    # s.range_multiple = 1.0
    s.area_min = 20
    s.area_multiple = 10000000
    s.range_multiple = 1.0

    disabled = True
    restore = 17 # 8
    load = False
    i_start = 0
    #i_start = 10764  # HELL BEGINS AT 10764

    # PRUNING:
    s.pruner = pruning.Pruner(N_scan=3, ratio_pruning=1e10, K_best=300)

    # DATASET:
    folder = '2018-06-01 2018-06-07'
    #folder = '2018-05-31'
    dataset = DatasetFolder(ROOT_FOLDER, folder)
    dt = dataset.config.dt

    # --- General parameters:
    detect_data = DetectData(dataset.config)
    area = detect_data.area()
    s.clutter_density = s.clutter_per_scan/area
    s.target_density = s.targets_per_scan/area

    # --- Boundary:
    meas_initializer = MeasInitBase(s.target_density, s.v_max)

    detector = Detector(dataset, detect_data, s.area_min, s.area_multiple, s.range_multiple)
    data_loader = dataloader.SimpleDataloader(detector)
    meas_model = MeasurementModel(s.q_std, s.r_std, dt, s.P_D, s.P_X, vmax=s.v_max)
    track_gate = TrackGate2(s.PG)

    # ---------- Printing info:
    print("Clutter/target density: {:.2g}/{:.2g}".format(s.clutter_density, s.target_density))
    print(s)
    print(len(dataset))

    mht_data = MhtData(dataset, settings=s)
    mht = MHT(dt, s.clutter_density, meas_model, track_gate, meas_initializer, mht_data, s.pruner)

    # ---------- Restoring data:
    if restore > -1:
        try:
            t = timer.SimpleTimer("Restore")
            mht_data.restore(8)
            t.report()
        except FileNotFoundError:
            pass

    # --------- The loader for MHT:
    mht_loader = dataloader.MHTLoader(mht, data_loader, i_start, disabled=disabled)
    if load:
        mht_loader[28850], mht_data.save()
        #mht_loader[len(dataset) - 1], mht_data.save()

    # --------- Start app:
    bg = background.createBackground(detect_data, None)
    app = QApplication(sys.argv)
    #ex = DynApp(data_loader, bg, i_start, True)
    track_gate = None
    ex = TrackApp(data_loader, bg, mht_loader, meas_model, track_gate, i_start, True)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    # set_excepthook()
    sys.setrecursionlimit(sys.getrecursionlimit() * 100000)
    # threading.stack_size(201326592)  # 64*3 MB
    # thread = threading.Thread(daemon=True, target=lambda: main())
    # thread.start()
    # thread.join()