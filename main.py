from PyQt5.QtWidgets import QApplication

from detection import dataset, detection
import dataloader
from gui.appmht import TrackApp
from mht import *
from tracking import *
from mhtdata import MhtData
import sys
from measinit import meas_init, boundary
from util import readMask, readImg
import gui.background as bg


def my_excepthook(type, value, tback):
    # log the exception here
    # then call the default handler
    sys.__excepthook__(type, value, tback)


def createBackground(range, landsat_mask, landsat_mask_extent):
    close_range = (-150, 130, -150, 20)
    long_range = (-175, 175, -175, 50)
    limits = close_range

    #landsat_mask, new_range = util.crop_square(~landsat_mask, range=190)

    background = bg.Background(range, limits[:2], limits[2:], out_size=1000)
    background.add_image(readImg("satellite/190.jpg"), 190)
    background.add_overlay(readMask("land/landAvg.png"), color=(1, 0, 1, 0.2), extent=175)
    background.add_overlay(landsat_mask, color=(0, 1, 0, 0.2), extent=landsat_mask_extent)
    background.add_overlay(readMask("land/landSat190impassable.png"), color=(1, 1, 1, 0.7), extent=190)
    return background


def create_land_mask(radar_range):
    landsat_mask = readMask("land/landSat190ext.png")
    range_mask = util.get_range_mask(190, radar_range, landsat_mask.shape[0])
    landsat_mask[range_mask] = True

    #land_mask, new_range = util.crop_square(~landsat_mask, 190)
    #new_extent = (-new_range, new_range, -new_range, new_range)
    landsat_mask, new_extent = util.shrink_mask(~landsat_mask, extent=(-190, 190, -190, 190))
    return landsat_mask, new_extent

if __name__ == '__main__':
    sys.excepthook = my_excepthook  # Traceback when using pyQT.

    # PARAMETERS:
    dt = 20
    q = 0.1758*1.5  # 0.1758
    r = 1/3
    v_max = 5  # 3.75
    n_init_std = 3  # Number of std. dev. to allow next measurement after init.
    PD = 0.7  # 0.956
    PX = (1-PD)*0.1
    PG = 0.95
    targets_per_scan = 0.02
    clutter_per_scan = 0.05
    meas_init_detections = 3
    radar_range = 175

    # --- General parameters:
    detect_data = detection.DetectData()

    # --- Boundary:
    boundary = boundary.setupBoundary(detect_data, v_max)
    meas_initializer = meas_init.MeasInit(boundary, dt, PD, meas_init_detections, targets_per_scan, v_max, init_speed=True)

    # --- Tracking parameters:
    clutter_density = clutter_per_scan/boundary.area()

    dataset = dataset.Dataset("radardata")
    detector = detection.Detection(dataset, detect_data, 0.5)
    dl = dataloader.SimpleDataloader(detector)

    # F and Q
    F = np.identity(4)
    F[0, 1] = dt
    F[2, 3] = dt

    Q = np.array([[dt**3 / 3, dt**2 / 2, 0, 0],
                  [dt**2 / 2, dt, 0, 0],
                  [0, 0, dt**3/3, dt**2/2],
                  [0, 0, dt**2/2, dt]])*(q**2)

    R = r*np.identity(2)  # (1/dt*0.5)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    land_mask, land_extent = create_land_mask(radar_range)
    background = createBackground(radar_range, ~land_mask, land_extent)

    #meas_model = MeasurementModel(H, R, F, Q, dt, PD, PX)
    meas_model = ProbMeasModel(H, R, F, Q, dt, PD, PX, land_mask, land_extent)
    track_gate = TrackGate2(PG)

    i_start = 27  # 27, 118

    mht_data = MhtData()
    mht = MHT(PD, dt, clutter_density, meas_model, track_gate, meas_initializer, mht_data)
    mht_loader = dataloader.MHTLoader(mht, dl, i_start, disabled=False)

    # Start app:
    app = QApplication(sys.argv)
    ex = TrackApp(dl, background, mht_loader, meas_model, track_gate, i=i_start)
    sys.exit(app.exec_())