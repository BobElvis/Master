from PyQt5.QtWidgets import QApplication

from detection.dataset import *
from detection.dataconfigs import *
from detection.detection import Detection, get_avg_mask, get_land_mask
import dataloader
from gui.appmht import TrackApp, DynApp
from tracking import *
from mht.mhtdata import MhtData
import sys
from measinit import meas_init, boundary
from util import readMask, readImg, set_excepthook, create_range_mask, shrink_mask
import gui.background as bg
import mht.pruning as pruning
import mht.mht as mht


def createBackground(data_config: DataConfig, landsat_mask, landsat_mask_extent):
    close_range = (-150, 130, -150, 20)
    long_range = (-175, 175, -175, 50)
    limits = close_range

    #landsat_mask, new_range = util.crop_square(~landsat_mask, range=190)

    background = bg.Background(range, limits[:2], limits[2:], out_size=1000)
    background.add_image(readImg("satellite/190.jpg"), 190)
    background.add_overlay(get_avg_mask(data_config), color=(1, 0, 1, 0.2), extent=data_config.radar_range)
    background.add_overlay(landsat_mask, color=(0, 1, 0, 0.2), extent=landsat_mask_extent)
    background.add_overlay(readMask("land/landSat190impassable.png"), color=(1, 1, 1, 0.7), extent=190)
    return background


def create_land_mask(data_config: DataConfig):
    landsat_mask = get_land_mask(data_config)
    range_mask = create_range_mask(190, data_config.radar_range, landsat_mask.shape[0])
    landsat_mask[range_mask] = True
    landsat_mask, new_extent = shrink_mask(~landsat_mask, extent=(-190, 190, -190, 190))
    return landsat_mask, new_extent


if __name__ == '__main__':
    set_excepthook()

    # PARAMETERS:
    q_std = 0.1758*2  # 0.1758
    r_std = 3/3  # 99.8 % lies withing +-3 meters.
    v_max = 5  # 3.75
    PG = 0.95
    targets_per_scan = 0.02
    clutter_per_scan = 0.05  # 0.05
    meas_init_detections = 3
    radar_range = 200

    # Calculate PD, PO, PX. Ratio PD/PO preserved.
    PD_base = 0.8  # 0.956
    PX_ratio = 0.2 #0.1
    PD = PD_base*(1-PX_ratio)
    PX = PX_ratio

    # DATASET:
    day = '2018-05-28'
    partition = 0
    dataset = Dataset(ROOT_FOLDER, day, partition)
    dt = dataset.config.dt

    # --- General parameters:
    detect_data = detection.detection.DetectData(dataset.config)
    pruner = pruning.Pruner(N_scan=-1, ratio_pruning=1e5, K_best=500)

    # --- Boundary:
    boundary = boundary.setupBoundary(detect_data, v_max)
    meas_initializer = meas_init.MeasInitBase(boundary.area(), targets_per_scan, v_max)
    #meas_initializer = meas_init.MeasInit(boundary, dt, PD_base, meas_init_detections, targets_per_scan, v_max, init_speed=True)

    # --- Tracking parameters:
    clutter_density = clutter_per_scan/boundary.area()


    land_mask = get_land_mask(dataset.config)
    detector = Detection(dataset, detect_data, 0.5)
    dl = dataloader.SimpleDataloader(detector)

    land_mask, land_extent = create_land_mask(dataset.config)
    background = createBackground(dataset.config, land_mask, land_extent)

    #meas_model = MeasurementModel(H, R, F, Q, dt, PD, PX)
    meas_model = ProbMeasModel(q_std, r_std, dt, PD, PX, land_mask, land_extent)
    track_gate = TrackGate2(PG)

    i_start = 0  # 27, 118

    mht_data = MhtData()
    mht = mht.MHT(dt, clutter_density, meas_model, track_gate, meas_initializer, mht_data, pruner)
    mht_loader = dataloader.MHTLoader(mht, dl, i_start, disabled=True)

    # Start app:
    app = QApplication(sys.argv)
    ex = DynApp(dl, background, i_start, True)
    #ex = TrackApp(dl, background, mht_loader, meas_model, track_gate, i=i_start)
    sys.exit(app.exec_())