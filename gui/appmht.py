from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from gui.app import *

ALL_CLUSTERS = -2
ALL_ALIVE = -1


class TrackApp(DynApp):
    def __init__(self, dataloader, background, mht_loader, meas_model, track_gate, i=0):
        self.num_hyps = 200
        self.num_clusters = 200

        self.meas_model = meas_model
        self.track_gate = track_gate

        self.i_mht_start = i
        self.i_mht_end = i
        self.mht_idx_start_edit = QLineEdit()
        self.mht_idx_last_edit = QLineEdit()

        self.allButton = QPushButton("All Clusters")
        self.allAliveButton = QPushButton("All Alive Clusters")
        self.layoutTrack = QVBoxLayout()
        self.listClusters = QListWidget()
        self.listHyp = QListWidget()
        self.mht_loader = mht_loader

        self.clusters = []
        self.selected_hyp_idx = 0
        self.selected_cluster_idx = ALL_ALIVE

        super().__init__(dataloader, background, i=i)

    def initUI(self):
        super().initUI()
        # self.listClusters.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.layoutContent.addLayout(self.layoutTrack)
        self.layoutTrack.addWidget(self.allButton)
        self.layoutTrack.addWidget(self.allAliveButton)
        self.layoutTrack.addWidget(self.listClusters)
        self.layoutTrack.addWidget(self.listHyp)

        self.layoutToolbar.addWidget(self.mht_idx_start_edit)
        self.layoutToolbar.addWidget(self.mht_idx_last_edit)
        self.mht_idx_start_edit.setFixedWidth(self.idx_edit_width)
        self.mht_idx_last_edit.setFixedWidth(self.idx_edit_width)

        self.listClusters.setSelectionMode(QListWidget.ExtendedSelection)
        self.listClusters.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.listHyp.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def __createMainView__(self, background):
        return ComboViewTrack(background, self.meas_model, self.track_gate)

    def setListeners(self):
        super().setListeners()
        self.listClusters.currentItemChanged.connect(self.on_cluster_changed)
        self.listHyp.currentItemChanged.connect(self.on_hyp_changed)

        self.allButton.clicked.connect(self.onAllClicked)
        self.allAliveButton.clicked.connect(self.onAllAliveClicked)

        self.mht_idx_start_edit.editingFinished.connect(self.onMhtIdxEdit)
        self.mht_idx_last_edit.editingFinished.connect(self.onMhtFirstIdxEdit)

    def onAllClicked(self):
        self.selected_cluster_idx = ALL_CLUSTERS
        self.update_cluster_list()
        self.update_hyp_list()
        self.update_track_data()

    def onAllAliveClicked(self):
        self.selected_cluster_idx = ALL_ALIVE
        self.update_cluster_list()
        self.update_hyp_list()
        self.update_track_data()

    @staticmethod
    def get_tracks(hyp):
        return [t.getTrack() for t in hyp.track_nodes], \
               [t.getTrack() for t in hyp.track_nodes_del]

    def update_mht_view(self):
        self.mht_idx_start_edit.setText(str(self.i_mht_start))
        self.mht_idx_last_edit.setText(str(self.i_mht_end))

        self.update_cluster_list()
        self.update_hyp_list()
        self.update_track_data()

    def __setup_data__(self, i):
        data = self.mht_loader[i]
        if data is None:
            self.clusters = []
        else:
            dead_clusters, alive_clusters = self.mht_loader[i]
            self.clusters = [(c, False) for c in dead_clusters] + [(c, True) for c in alive_clusters]

    def on_idx_changed(self, curr_i, prev_i):
        # Mht-idx:
        self.i_mht_end = curr_i if self.i_mht_end == prev_i else self.i_mht_end

        # Load mht.
        if prev_i is not None and prev_i + 1 < curr_i:
            for i in range(prev_i + 1, curr_i + 1):
                self.idxEdit.setText("{} ->".format(i))
                self.main_view.set_scan(self.dataloader[i])
                self.main_view.update_view()
                self.__setup_data__(i)
                self.update_mht_view()
        else:
            self.__setup_data__(curr_i)

        super().on_idx_changed(curr_i, prev_i)
        self.update_mht_view()

    def update_cluster_list(self):
        print("Updating cluster list.")
        self.listClusters.clear()

        n_clusters = min(len(self.clusters), self.num_clusters)
        for idx in range(n_clusters):
            cluster, alive = self.clusters[idx]
            item = QListWidgetItem('{}: NTGT={}'.
                                   format(idx + 1, len(cluster.targets)))
            item.setData(Qt.UserRole, idx)
            self.listClusters.addItem(item)
            if self.selected_cluster_idx == ALL_CLUSTERS:
                selected = True
            elif self.selected_cluster_idx == ALL_ALIVE:
                selected = alive
            else:
                selected = idx == self.selected_cluster_idx
            item.setSelected(selected)

    def update_hyp_list(self):
        print("Updating hyp list")
        self.listHyp.clear()

        if self.selected_cluster_idx < 0:
            return

        idx = min(self.selected_cluster_idx, len(self.clusters))
        cluster = self.clusters[idx][0]
        n_hyp = min(self.num_hyps, len(cluster.leaves))
        self.selected_hyp_idx = 0
        for i in range(0, n_hyp):
            item = QListWidgetItem('{}: {:.4g}%'.format(i + 1, cluster.leaves[i].probability * 100))
            item.setData(Qt.UserRole, i)
            self.listHyp.addItem(item)
            if i == self.selected_hyp_idx:
                item.setSelected(True)

    def update_track_data(self):
        print("Updating track data.")
        if self.selected_cluster_idx >= 0:
            cluster, _ = self.clusters[self.selected_cluster_idx]
            tracks, del_tracks = self.get_tracks(cluster.leaves[self.selected_hyp_idx])
        else:
            tracks = []
            del_tracks = []
            for cluster, alive in self.clusters:
                if self.selected_cluster_idx == ALL_ALIVE and not alive:
                    continue
                alive_track, del_track = self.get_tracks(cluster.leaves[0])
                tracks.extend(alive_track)
                del_tracks.extend(del_track)
        self.main_view.radarFig.set_track_data(tracks, del_tracks)
        self.main_view.update_view()

    def on_cluster_changed(self, curr, prev):
        if curr is None:
            return
        i = curr.data(Qt.UserRole)
        self.selected_cluster_idx = i
        self.update_hyp_list()
        self.update_track_data()

    def on_hyp_changed(self, curr, prev):
        self.selected_hyp_idx = curr.data(Qt.UserRole)
        self.update_track_data()

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