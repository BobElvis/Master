from PyQt5.QtWidgets import QApplication
from gui.appmht import TrackApp, DynApp

from detection.dataset import *
from detection.dataconfigs import *
from detection.detection import Detection, get_avg_mask, get_land_mask, DetectData
import dataloader
from tracking import *
from mht.mhtdata import MhtData
import sys
from measinit import meas_init, boundary
from util import readMask, readImg, set_excepthook, create_range_mask, shrink_mask
import gui.background as bg
import mht.pruning as pruning
import mht.mht as mht

def createBackground(detect_data: DetectData, boundary_img):
    close_range = (-150, 130, -150, 20)
    long_range = (-175, 175, -175, 50)
    limits = long_range

    data_config = detect_data.data_config
    background = bg.Background(data_config.radar_range, limits[:2], limits[2:], out_size=1024)
    background.add_image(readImg("satellite/190.jpg"), 190)
    background.add_overlay(~detect_data.sat_mask, color=(0, 1, 0, 0.2), extent=data_config.radar_range)
    background.add_overlay(detect_data.avg_mask, color=(1, 0, 1, 0.2), extent=data_config.radar_range)
    background.add_image(boundary_img, extent=data_config.radar_range)
    #background.add_overlay(readMask("land/landSat190impassable.png"), color=(1, 1, 1, 0.7), extent=190)
    return background


def create_land_mask(data_config: DataConfig):
    landsat_mask = get_land_mask(data_config)
    range_mask = create_range_mask(190, data_config.radar_range, landsat_mask.shape[0])
    ext = max(190, data_config.radar_range)
    landsat_mask[range_mask] = True
    landsat_mask, new_extent = shrink_mask(~landsat_mask, extent=(-ext, ext, -ext, ext))
    return landsat_mask, new_extent


if __name__ == '__main__':
    set_excepthook()

    # PARAMETERS:
    q_std = 0.1758*1.5  # 0.1758
    r_std = 6/3  # 99.8 % lies withing +-6 meters.
    v_max = 5  # 3.75
    PG = 0.95
    targets_per_scan = 0.001
    clutter_per_scan = 2  # 0.05
    meas_init_detections = 3
    radar_range = 200
    P_D = 0.8  # Not considering deletion.
    P_X = 0.15
    print("PD: {:.2f}, PX: {:.2f}, PO: {:.2f}".format(P_D, P_X, 1 - P_D - P_X))

    # PRUNING:
    pruner = pruning.Pruner(N_scan=-1, ratio_pruning=1e7, K_best=500)

    # DATASET:
    day = '2018-05-30'
    partition = 0
    dataset = Dataset(ROOT_FOLDER, day, partition)
    dt = dataset.config.dt
    i_start = 167  # 27, 118

    # --- General parameters:
    detect_data = detection.detection.DetectData(dataset.config)
    boundary, boundary_img = boundary.create_boundary(detect_data)
    clutter_density = clutter_per_scan/boundary.area()

    # --- Boundary:
    meas_initializer = meas_init.MeasInitBase(boundary.area(), targets_per_scan, v_max)
    #meas_initializer = meas_init.MeasInitProb(boundary, dt, P_D_base, meas_init_detections, targets_per_scan, v_max)

    detector = Detection(dataset, detect_data, 0.5)
    dl = dataloader.SimpleDataloader(detector)

    background = createBackground(detect_data, boundary_img)

    meas_model = MeasurementModel(q_std, r_std, dt, P_D, P_X)
    #meas_model = ProbMeasModel(q_std, r_std, dt, PD, PX, land_mask, land_extent)
    track_gate = TrackGate2(PG)

    mht_data = MhtData(dataset)
    mht = mht.MHT(dt, clutter_density, meas_model, track_gate, meas_initializer, mht_data, pruner)
    mht_loader = dataloader.MHTLoader(mht, dl, i_start, disabled=False)

    # Printing info:
    print(boundary)
    print("Clutter/target ratio: {}".format(clutter_per_scan/targets_per_scan))

    # Start app:
    app = QApplication(sys.argv)
    #ex = DynApp(dl, background, i_start, True)
    ex = TrackApp(dl, background, mht_loader, meas_model, track_gate, i_start, True)
    sys.exit(app.exec_())