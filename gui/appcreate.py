from gui.app import *
from gui.appmht import ComboViewTrack
from dataset import DatasetFolder
from dataloader import BaseLoader
from tracking import MeasurementModel
from mht.mht import MHT
from model.model import Measurement, TrackNode, Detection
from util import get_filename
from dataset import ROOT_FOLDER_TRUTH
import pickle
import sys


def center_detections(detections):
    tot_area = 0
    pos = np.array((0, 0), dtype=np.float64)
    for detection in detections:
        tot_area += detection.area
        pos += detection.pos * detection.area
    pos = pos / tot_area
    return Detection(pos, tot_area, None)


class MarkData:
    def __init__(self, dataset: DatasetFolder, dataloader: BaseLoader, meas_model: MeasurementModel):
        self.dataset = dataset
        self.meas_model = meas_model
        self.dataloader = dataloader

        self.target_idx = 0
        self.data = []  # [scan_idx]{target_idx} -> tuple of meas idx
        self.scan_meas_idx_map = []  # [scan_idx][target_idx,...]
        self.reverse_map = dict()  # target_idx -> (scan_idx, scan_meas_idx, detection)

        # Constants:
        self.init_speed = np.array([0, 0])
        self.init_speed_var = np.array([1 / 3, 1 / 3]) * 7

    def allocate(self, to_idx):
        if to_idx < len(self.data):
            return

        prev_items = [] if to_idx == 0 else \
            [key for key, value in self.data[len(self.data) - 1].items() if value is not None]

        for scan_idx in range(len(self.data), to_idx + 1):
            scan = self.dataloader[scan_idx]
            n = len(scan)

            # Create mapping for indices:
            idx_list = []
            for j, detection in enumerate(scan.detections):
                meas_idx = j + self.target_idx
                idx_list.append(meas_idx)
                self.reverse_map[meas_idx] = (scan_idx, j, detection)
            self.scan_meas_idx_map.append(idx_list)
            self.target_idx += n

            new_dict = dict.fromkeys(prev_items, ())
            self.data.append(new_dict)
            prev_items = [key for key, value in self.data[scan_idx].items() if value is not None]

    def shift_up(self, scan_idx, prev_source_idx, keep_rest=True):

        # Find the next assignment for the track. If two, choose the first.
        new_source_idx = None
        for i in range(scan_idx + 1, len(self.data)):
            data_scan = self.data[i]
            val = data_scan[prev_source_idx]
            if val is None:  # The track is deleted here.
                break
            del data_scan[prev_source_idx]  # Delete previous entry
            if new_source_idx is None:
                if len(val) > 0:
                    # This is the next source of the track
                    new_source_idx = val[0]
                    data_scan[new_source_idx] = (new_source_idx,)
                # Not detected.
            elif keep_rest:
                # Set the track source:
                data_scan[new_source_idx] = val
        if new_source_idx is None:
            print(" - The track does not continue")

    def set_meas(self, scan_idx, meas_idx, new):

        # Determine if changed:
        source_idx = self.scan_meas_idx_map[scan_idx][meas_idx]
        print("Setting meas {} ({}) at scan {}".format(meas_idx, source_idx, scan_idx))
        data_scan = self.data[scan_idx]
        prev_new = source_idx in data_scan
        if prev_new == new:
            return False

        if new:
            # Add track:
            self.data[scan_idx][source_idx] = (source_idx,)

            # TODO: Quite intensive for large data. Consider "max track length"
            for i in range(scan_idx + 1, len(self.data)):
                self.data[i][source_idx] = ()
        else:
            del self.data[scan_idx][source_idx]
            self.shift_up(scan_idx, source_idx, keep_rest=True)

        return True

    def set_target(self, scan_idx, target_idx, measurements):
        # measurements are idx of measurements, each in [0, n]
        data_scan = self.data[scan_idx]
        prev_value = data_scan[target_idx]

        # Check if deletion:
        if measurements is None:
            if prev_value is None:
                return False
            else:
                print(" - Deleting rest of track")
                self.data[scan_idx][target_idx] = measurements
                self.shift_up(scan_idx, target_idx, keep_rest=False)
                return True

        # No deletion:
        new_measurements = tuple([self.scan_meas_idx_map[scan_idx][m] for m in measurements])
        if new_measurements == prev_value:
            return False
        else:
            if prev_value is None:
                # TODO: Quite intensive for large data. Consider "max track length"
                for i in range(scan_idx + 1, len(self.data)):
                    self.data[i][target_idx] = set()
            print("Setting target {} at scan {} with {}".format(target_idx, scan_idx, new_measurements))
            data_scan[target_idx] = new_measurements
        return True

    def get_new_targets(self, scan_idx):
        self.allocate(scan_idx)
        data_scan = self.data[scan_idx]
        source_ind = self.scan_meas_idx_map[scan_idx]
        values = []
        for source_idx in source_ind:
            values.append(source_idx in data_scan)
        return values

    def get_existing(self, idx):
        self.allocate(idx)
        values = []
        for key, value in self.data[idx].items():
            if value is None:
                values.append((key, None))
            elif key not in value:
                mapping = [self.reverse_map[v][1] for v in value]
                values.append((key, mapping))
        values = sorted(values, key=lambda x: x[0])
        return values

    ###############################
    # GET DATA
    ###############################

    def get_tracks_scan(self, scan_idx, smooth=True):
        deleted = []
        alive = []

        for source_idx, val in self.data[scan_idx].items():
            track_nodes = self.get_track_nodes(source_idx)
            # track = track_nodes
            track = track_nodes[-1].getTrack(i_end=scan_idx)

            if val is None:
                deleted.append(track)
            else:
                alive.append(track)
        if smooth:
            deleted = [self.meas_model.RTS(track) for track in deleted]
        return alive, deleted

    def get_meas_idx_track(self, source_idx):
        first_scan_idx = self.reverse_map[source_idx][0]
        measurements = []
        for i in range(first_scan_idx, len(self.data)):
            meas_ind = self.data[i][source_idx]
            if meas_ind is None:
                break
            else:
                measurements.append(meas_ind)
        return measurements, first_scan_idx

    def get_detection_track(self, source_idx):
        meas_idx_track, first_scan_idx = self.get_meas_idx_track(source_idx)
        detections = [self.get_detections(indices) for indices in meas_idx_track]
        return detections, first_scan_idx

    def get_detection_track_with_idx(self, source_idx):
        meas_idx_track, first_scan_idx = self.get_meas_idx_track(source_idx)
        detection_track = [[(self.get_detection(idx), idx) for idx in indices] for indices in meas_idx_track]
        return detection_track, first_scan_idx

    def get_track_nodes(self, source_idx):
        detections, first_scan_idx = self.get_detection_track(source_idx)
        nodes = self.create_track(detections, first_scan_idx)
        return nodes

    def create_track(self, detection_track, first_scan_idx):
        iterator = iter(detection_track)

        # Create first:
        detection = next(iterator)[0]

        mean, covariance = self.meas_model.initialize(detection)
        first_node = TrackNode(first_scan_idx, None, mean, covariance, None, detection)
        self.meas_model.predict(first_node)

        # Create remaining:
        nodes = [first_node]
        for i, detections in enumerate(iterator):
            if len(detections) == 0:
                meas = None
            elif len(detections) == 1:
                meas = detections[0]
            else:
                meas = center_detections(detections)
            new_node = nodes[i].innovateTrack(meas, self.meas_model)
            self.meas_model.predict(new_node)
            nodes.append(new_node)
        assert len(nodes) == len(detection_track)
        return nodes

    def get_detection(self, idx):
        return self.reverse_map[idx][2]

    def get_detections(self, indices):
        return [self.get_detection(idx) for idx in indices]

    #################################
    # SAVING
    #################################

    def save(self):
        filename = get_filename(ROOT_FOLDER_TRUTH, str(self.dataset), True)
        util.save_data((self.data, self.scan_meas_idx_map, self.reverse_map, self.target_idx), filename)

    def load(self, name=None):
        name = str(self.dataset) if name is None else name
        filename = get_filename(ROOT_FOLDER_TRUTH, name, False)
        self.data, self.scan_meas_idx_map, self.reverse_map, self.target_idx = util.load_data(filename)

    def get_last_scan_idx(self):
        return len(self.data) - 1


class AppCreate(DynApp):
    def __init__(self, dataloader: BaseLoader, mark_data: MarkData, background, meas_model, gate=None, i=0):
        self.measurement_data = mark_data
        self.table_new = QTableWidget()
        self.table_target = QTableWidget()
        self.saveButton = QPushButton("Save")
        self.updating_table = False
        self.meas_model = meas_model
        self.gate = gate
        super().__init__(dataloader, background, i=i)

    def initUI(self):
        super().initUI()
        self.table_new.horizontalHeader().setVisible(False)
        self.table_new.verticalHeader().setVisible(False)
        self.table_new.setColumnCount(1)
        self.table_new.setRowCount(0)
        self.table_new.setMaximumWidth(150)
        self.table_new.itemClicked.connect(self.handleItemClickedNew)

        self.table_target.horizontalHeader().setVisible(False)
        self.table_target.verticalHeader().setVisible(False)
        self.table_target.setColumnCount(2)
        self.table_target.setRowCount(0)
        self.table_target.setMaximumWidth(400)
        self.table_target.itemChanged.connect(self.handleTargetChanged)
        self.table_target.itemClicked.connect(self.clickItem)

        self.layoutToolbar.addWidget(self.saveButton)
        self.saveButton.clicked.connect(self.onSaveClicked)

        self.layoutContent.addWidget(self.table_new)
        self.layoutContent.addWidget(self.table_target)
        self.update_table_data(self.i)

    def on_idx_changed(self, curr_i, prev_i):
        self.update_table_data(curr_i)
        self.update_track_data()
        super().on_idx_changed(curr_i, prev_i)

    def update_table_data(self, idx):
        self.updating_table = True
        meas_values = self.measurement_data.get_new_targets(idx)
        track_map = self.measurement_data.get_existing(idx)

        scan = self.dataloader[idx]

        self.table_new.setRowCount(len(meas_values))
        self.table_target.setRowCount(len(track_map))

        for idx in range(len(scan)):
            item = QTableWidgetItem("{}".format(idx + 1))
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

            item.setCheckState(Qt.Checked if meas_values[idx] else Qt.Unchecked)
            self.table_new.setItem(idx, 0, item)

        for idx, (target_idx, measurements) in enumerate(track_map):
            if measurements is None:
                string = "D"
            else:
                string = ""
                for m in measurements:
                    string += "{} ".format(m+1)

            item = QTableWidgetItem("{}".format(target_idx))
            item.setFlags(Qt.ItemIsEnabled)

            self.table_target.setItem(idx, 0, item)
            self.table_target.setItem(idx, 1, QTableWidgetItem(string))
        self.updating_table = False

    def update_track_data(self):
        alive, deleted = self.measurement_data.get_tracks_scan(self.i)
        self.main_view.radarFig.set_track_data(alive, deleted)
        # print("Alive: {}".format(alive))
        # print("Deleted: {}".format(deleted))

    def handleItemClickedNew(self, item):
        if self.updating_table:
            return
        self.measurement_data.set_meas(self.i, int(item.text()) - 1, item.checkState() == Qt.Checked)

    def handleTargetChanged(self, item):
        if self.updating_table:
            return

        target_idx = int(self.table_target.item(item.row(), 0).text())
        cell_text = item.text()
        error = False

        if cell_text == "D":
            meas_indices = None
        else:
            max_n = len(self.dataloader[self.i])
            meas_indices_text = cell_text.split()
            try:
                meas_indices = tuple([int(meas_idx) - 1 for meas_idx in meas_indices_text
                                      if int(meas_idx) <= max_n])
            except ValueError:
                print("WARNING: The values ({}) for target {} was not valid.".format(cell_text, target_idx))
                error = True

        if not error:
            changed = self.measurement_data.set_target(self.i, target_idx, meas_indices)
        else:
            changed = False
            self.update_table_data(self.i)

        if changed:
            self.update_track_data()
            self.draw(now=True)

    def __createMainView__(self, background):
        return ComboViewTrack(background, self.meas_model, self.gate)

    def onSaveClicked(self):
        self.measurement_data.save()

    def clickItem(self, item):
        self.table_target.editItem(item)
