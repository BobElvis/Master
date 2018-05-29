import dataloader
import pysimulator.scenarios.scenarios
from model.model import *
import pysimulator.scenarios.defaults as defaults
import pysimulator.simulationConfig as config
import gui.background as bg
from tracking import *
import mht.mht as mht
from mht import mhtdata
from PyQt5.QtWidgets import QApplication
from gui.appmht import TrackApp
import sys
import measinit.meas_init as mi
import pymht.models.constants as constants
import mht.pruning as pruning


class SimulatorSource(dataloader.DataSource):
    def __init__(self, scenario, clutter_density, P_D, seed=0):
        self.scenario = scenario
        simList = scenario.getSimList()

        # List of Measurement List:
        self.scanList, _ = scenario.getSimulatedScenario(seed, simList, clutter_density, P_D, localClutter=False)

    def load_data(self, idx) -> Scan:
        measurements = self.scanList[idx].measurements
        #radar_image = np.zeros((200, 200, 4))
        #cam_image = np.zeros((200, 400, 3))
        radar_image, cam_image = None, None
        return Scan(0, radar_image, cam_image, measurements[:, 0], measurements[:, 1])

    def __len__(self):
        return len(self.scanList)


def simulator_background(range):
    import util
    background = bg.Background(range, (-range, range), (-range, range), out_size=1000)
    background.add_image(np.ones((1000, 1000, 3)), range)
    background.add_overlay(~util.create_range_mask(range, range, 1000), (0, 1, 0, 1), range)
    return background

def my_excepthook(type, value, tback):
    # log the exception here
    # then call the default handler
    sys.__excepthook__(type, value, tback)

if __name__ == '__main__':
    sys.excepthook = my_excepthook
    scenario = pysimulator.scenarios.scenarios.scenario0
    print("Number of targets in scenario: {}".format(len(scenario.initialTargets)))

    # PARAMETERS:
    dt = scenario.radarPeriod
    q_std = 2 #0.1758*2  # 0.1758
    r_std = constants.sigmaR_RADAR_true  # 99.8 % lies withing +-3 meters.
    v_max = defaults.maxSpeedMS
    PG = 0.95
    P_X = 0
    targets_per_scan = 1
    clutter_density = config.lambdaphiList[2]  # 0.05
    radar_range = scenario.radarRange

    pruner = pruning.Pruner(N_scan=-1, ratio_pruning=1e5, K_best=20)

    # Calculate PD, PO, PX. Ratio PD/PO preserved.
    PD_base = 0.8  # 0.956
    PX_ratio = 0  # 0.1
    PD = PD_base * (1 - PX_ratio)
    PX = PX_ratio

    # DERIVED:
    area = np.pi * (radar_range**2)
    scenarioSource = SimulatorSource(scenario, clutter_density, PD_base)
    dl = dataloader.SimpleDataloader(scenarioSource)
    background = simulator_background(radar_range)
    meas_model = MeasurementModel(q_std, r_std, dt, PD, P_X)
    #meas_model = ProbMeasModel(q_std, r_std, dt, PD, PX, land_mask, land_extent)
    track_gate = TrackGate2(PG)
    meas_initializer = mi.MeasInitBase(area, targets_per_scan, v_max)

    i_start = 0  # 27, 118

    mht_data = mhtdata.MhtData()
    mht = mht.MHT(dt, clutter_density, meas_model, track_gate, meas_initializer, mht_data, pruner)
    mht_loader = dataloader.MHTLoader(mht, dl, i_start, disabled=False)

    # Start app:
    app = QApplication(sys.argv)
    ex = TrackApp(dl, background, mht_loader, meas_model, track_gate, i=i_start)
    sys.exit(app.exec_())
