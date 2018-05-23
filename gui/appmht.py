from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from gui.app import *


class TrackApp(DynApp):
    def __init__(self, dataloader, background, mht_loader, meas_model, track_gate, i=0):
        self.num_hyps = 200
        self.meas_model = meas_model
        self.track_gate = track_gate

        self.i_mht_start = i
        self.i_mht_end = i
        self.mht_idx_start_edit = QLineEdit()
        self.mht_idx_last_edit = QLineEdit()

        self.layoutTrack = QVBoxLayout()
        self.listClusters = QListWidget()
        self.listHyp = QListWidget()
        self.mht_loader = mht_loader

        self.clusters = None
        self.hypothesises = None
        self.updatePending = False

        super().__init__(dataloader, background, i=i)

    def initUI(self):
        super().initUI()
        # self.listClusters.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.layoutContent.addLayout(self.layoutTrack)
        self.layoutTrack.addWidget(self.listClusters)
        self.layoutTrack.addWidget(self.listHyp)

        self.layoutToolbar.addWidget(self.mht_idx_start_edit)
        self.layoutToolbar.addWidget(self.mht_idx_last_edit)
        self.mht_idx_start_edit.setFixedWidth(self.idx_edit_width)
        self.mht_idx_last_edit.setFixedWidth(self.idx_edit_width)

        self.listClusters.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.listHyp.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def __createMainView__(self, background):
        return ComboViewTrack(background, self.meas_model, self.track_gate)

    def update_view(self):
        hyps_list = self.mht_loader[self.i]

        if hyps_list is None or len(hyps_list) == 0:
            self.clusters = []
            self.hypothesises = []
        else:
            self.clusters = hyps_list
            self.hypothesises = hyps_list[0][0]

        self.updatePending = True
        prevCluster = self.listClusters.currentRow()
        self.updateClusterList(prevCluster)
        self.updateHypList()
        self.updatePending = False

    def on_idx_changed(self, curr_i, prev_i):
        self.update_view()

        # Mht-idx:
        self.i_mht_end = curr_i if self.i_mht_end == prev_i else self.i_mht_end
        self.mht_idx_start_edit.setText(str(self.i_mht_start))
        self.mht_idx_last_edit.setText(str(self.i_mht_end))

        super().on_idx_changed(curr_i, prev_i)

    def updateClusterList(self, cluster_i=None):
        self.listClusters.clear()
        for i in range(len(self.clusters)):
            item = QListWidgetItem('{}: NTGT={}'.format(i + 1, len(self.clusters[i][1])))
            item.setData(Qt.UserRole, i)
            self.listClusters.addItem(item)

        cluster_i = len(self.clusters)-1 if None else cluster_i
        if len(self.clusters) > 0:
            self.listClusters.setCurrentRow(min(len(self.clusters)-1, cluster_i))

    def updateHypList(self):
        self.listHyp.clear()
        n = min(self.num_hyps, len(self.hypothesises))
        for i in range(0, n):
            item = QListWidgetItem('{}: {:.4g}%'.format(i + 1, self.hypothesises[i].probability * 100))
            item.setData(Qt.UserRole, i)
            self.listHyp.addItem(item)
        if n > 0:
            self.listHyp.setCurrentRow(0)

    def setListeners(self):
        super().setListeners()
        self.listClusters.currentItemChanged.connect(self.on_cluster_changed)
        self.listHyp.currentItemChanged.connect(self.on_hyp_changed)
        self.mht_idx_start_edit.editingFinished.connect(self.onMhtIdxEdit)
        self.mht_idx_last_edit.editingFinished.connect(self.onMhtFirstIdxEdit)

    def set_track_data(self, hyp):
        if hyp is None:
            self.main_view.radarFig.set_track_data([], [])
            return
        tracks = [t.getTrack() for t in hyp.track_nodes]
        del_tracks = [t.getTrack() for t in hyp.track_nodes_del]
        self.main_view.radarFig.set_track_data(tracks, del_tracks)

    def on_cluster_changed(self, curr, prev):
        self.hypothesises = [] if curr is None else self.clusters[curr.data(Qt.UserRole)][0]
        self.updateHypList()

    def on_hyp_changed(self, curr, prev):
        hyp = None if curr is None else self.hypothesises[curr.data(Qt.UserRole)]
        self.set_track_data(hyp)
        if not self.updatePending:
            self.main_view.update_view()

    def onMhtIdxEdit(self):
        idx = int(float(self.mht_idx_start_edit.text()))

    def onMhtFirstIdxEdit(self):
        idx = int(float(self.mht_idx_last_edit.text()))


class ComboViewTrack(ComboView):
    def __init__(self, background, meas_model, track_gate):
        self.meas_model = meas_model
        self.track_gate = track_gate
        super().__init__(background)

    def __createRadarFig__(self, ax, background):
        return TrackFig(ax, background, self.meas_model, self.track_gate)