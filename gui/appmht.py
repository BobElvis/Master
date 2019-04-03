from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from gui.app import *
from gui.guiutil import get_idx
from dataloader import MHTLoader

_ALL_CLUSTERS = -2
_ALL_ALIVE = -1


class TrackApp(DynApp):
    def __init__(self, dataloader, background, mht_loader: MHTLoader, meas_model, track_gate, i=0, show_progess=False):
        self.num_hyps = 200
        self.num_clusters = 200
        self.show_progress = show_progess

        self.meas_model = meas_model
        self.track_gate = track_gate

        self.i_track_start = i
        self.i_mht = i
        self.mht_idx_start_edit = QLineEdit()
        self.mht_idx_edit = QLineEdit()

        self.allButton = QPushButton("All Clusters")
        self.allAliveButton = QPushButton("All Alive Clusters")
        self.saveButton = QPushButton("Save MHT")
        self.layoutTrack = QVBoxLayout()
        self.listClusters = QListWidget()
        self.listHyp = QListWidget()
        self.mht_loader = mht_loader

        self.clusters = []
        self.mode_all = False  # True: All, False: All_alive, None: Specific cluster

        self.selected_cluster_idx = None
        self.selected_hyp_idx = 0
        super().__init__(dataloader, background, i=i)
        self.update_mht_view(level=0)

    # SETUP:

    def initUI(self):
        super().initUI()
        # self.listClusters.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.layoutContent.addLayout(self.layoutTrack)
        self.layoutTrack.addWidget(self.allButton)
        self.layoutTrack.addWidget(self.allAliveButton)
        self.layoutTrack.addWidget(self.listClusters)
        self.layoutTrack.addWidget(self.listHyp)
        self.layoutTrack.addWidget(self.saveButton)

        self.listClusters.setMinimumWidth(500)
        self.layoutToolbar.addWidget(self.mht_idx_start_edit)
        self.layoutToolbar.addWidget(self.mht_idx_edit)
        self.mht_idx_start_edit.setFixedWidth(self.idx_edit_width)
        self.mht_idx_edit.setFixedWidth(self.idx_edit_width)

        self.listClusters.setSelectionMode(QListWidget.ExtendedSelection)
        self.listClusters.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.listHyp.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def __createMainView__(self, background):
        return ComboViewTrack(background, self.meas_model, self.track_gate)

    def setListeners(self):
        super().setListeners()
        self.connect(self.listClusters.currentItemChanged, self.on_cluster_changed)
        self.connect(self.listHyp.currentItemChanged, self.on_hyp_changed)
        self.connect(self.allButton.clicked, self.onAllClicked)
        self.connect(self.allAliveButton.clicked, self.onAllAliveClicked)
        self.connect(self.mht_idx_start_edit.editingFinished,
                     lambda: self.on_mht_first_idx_changed(get_idx(self.mht_idx_start_edit)))
        self.connect(self.mht_idx_edit.editingFinished,
                     lambda: self.on_mht_idx_changed(get_idx(self.mht_idx_edit)))
        self.connect(self.saveButton.clicked, self.onSaveClicked)

    # END SETUP

    # EVENT HANDLERS:

    def onSaveClicked(self):
        self.mht_loader.mht_data.save()

    def onAllClicked(self):
        self.mode_all = True
        self.update_cluster_idx()
        self.update_mht_view(level=0)

    def onAllAliveClicked(self):
        self.mode_all = False
        self.update_cluster_idx()
        self.update_mht_view(level=0)

    def on_idx_changed(self, curr_i, prev_i):
        #if self.i_track_start != 0:
        #    self.i_track_start += curr_i - prev_i

        # Viewing index is changing...
        if curr_i > self.i_mht or (abs(curr_i - prev_i) == 1 and self.i_mht == prev_i):
            self.i_mht = curr_i
            self.on_mht_idx_changed(curr_i)

        # Dependant on
        self.set_track_data()
        super().on_idx_changed(curr_i, prev_i)

    def on_mht_idx_changed(self, new_i):
        self.i_mht = min(len(self.dataloader) - 1, new_i)

        # Update data:
        data = self.mht_loader[self.i_mht]
        if data is None:
            self.clusters = []
        else:
            dead_clusters, alive_clusters = data
            self.clusters = dead_clusters + alive_clusters

        self.update_cluster_idx()

        self.update_mht_view(level=0)

    def on_mht_first_idx_changed(self, new_i):
        self.i_track_start = new_i
        self.update_mht_view(level=2)

    def on_cluster_changed(self, curr, prev):
        if curr is None:
            return
        i = curr.data(Qt.UserRole)
        self.mode_all = None
        self.selected_cluster_idx = i
        cluster = self.clusters[self.selected_cluster_idx]
        if cluster.last_idx is not None and not (cluster.first_idx < self.i < cluster.last_idx):
            self.change_idx(cluster.last_idx)
        self.update_mht_view(level=1)

    def on_hyp_changed(self, curr, prev):
        if curr is None:
            self.selected_hyp_idx = 0
        else:
            self.selected_hyp_idx = curr.data(Qt.UserRole)
        self.update_mht_view(level=2)

    # END EVENT HANDLERS.

    def update_mht_view(self, level=0):
        self.mht_idx_start_edit.setText(str(self.i_track_start))
        self.mht_idx_edit.setText(str(self.i_mht))

        if level <= 0:
            self.update_cluster_list()
        if level <= 1:
            self.update_hyp_list()
        self.set_track_data()
        self.draw(radar=True)

    def update_cluster_idx(self):
        # Update selected cluster idx:
        if self.isAllClusters() or self.isAllAliveClusters():
            selected_clusters = self.get_selected_clusters()
            #print(len(selected_clusters))
            if len(selected_clusters) == 1:
                self.selected_cluster_idx = len(self.clusters) - 1
            else:
                self.selected_cluster_idx = None
        else:
            self.selected_cluster_idx = min(self.selected_cluster_idx, len(self.clusters) - 1)
        #print(self.selected_cluster_idx)

    # Utility methods:

    def isAllAliveClusters(self):
        return (self.mode_all is not None) and (self.mode_all is False)

    def isAllClusters(self):
        return self.mode_all is not None and self.mode_all

    def get_tracks(self, hyp):
        tracks_alive = [t.getTrack(self.i_track_start, self.i) for t in hyp.track_nodes]
        tracks_del = [t.getTrack(self.i_track_start, self.i) for t in hyp.track_nodes_del]
        tracks = tracks_alive + tracks_del
        tracks_alive = [t for t in tracks if len(t) > 0 and t[-1].scan_idx == self.i]
        tracks_del = [t for t in tracks if len(t) > 0 and t[-1].scan_idx < self.i]
        return tracks_alive, tracks_del

    def get_selected_clusters(self):
        if self.isAllClusters():
            return self.clusters
        elif self.isAllAliveClusters():
            return [c for c in self.clusters if c.last_idx is None]
        else:
            return [self.clusters[self.selected_cluster_idx]]

    # END UTILITY..

    # START UPDATERS:

    def update_cluster_list(self):
        self.listClusters.clear()

        for idx, cluster in enumerate(self.clusters):
            alive = cluster.last_idx is None
            last_idx = "" if alive else cluster.last_idx
            best_hyp = cluster.leaves[0]
            if len(cluster.leaves) > 1:
                ratio = cluster.leaves[0].probability/cluster.leaves[1].probability
            else:
                ratio = 1e7
            targets = len(best_hyp.track_nodes) + len(best_hyp.track_nodes_del)
            ratio_str = "" if ratio > 10000 else "{:.3f}".format(ratio)
            target_str = "" if targets == 0 else "T={}".format(targets)

            desc = "{}: {}->{} | {} | {} ". \
                format(idx + 1, cluster.first_idx, last_idx, target_str, ratio_str)
            item = QListWidgetItem(desc)
            item.setData(Qt.UserRole, idx)
            self.listClusters.addItem(item)

            # Check if should be selected:
            if self.isAllClusters():
                selected = True
            elif self.isAllAliveClusters():
                selected = alive
            else:
                selected = idx == self.selected_cluster_idx
            item.setSelected(selected)

    def update_hyp_list(self):
        self.listHyp.clear()

        if self.selected_cluster_idx is None:
            return

        cluster = self.clusters[self.selected_cluster_idx]
        n_hyp = min(self.num_hyps, len(cluster.leaves))
        self.selected_hyp_idx = 0
        for i in range(0, n_hyp):
            node = cluster.leaves[i]
            item = QListWidgetItem('{}: {:.4g}% | {} | {:.2g}'.format(i + 1, node.probability * 100, node.K_max+1, node.ratio_max))
            item.setData(Qt.UserRole, i)
            self.listHyp.addItem(item)
            if i == self.selected_hyp_idx:
                item.setSelected(True)

    def set_track_data(self):
        if self.selected_cluster_idx is not None:
            cluster = self.clusters[self.selected_cluster_idx]
            tracks, del_tracks = self.get_tracks(cluster.leaves[self.selected_hyp_idx])
        else:
            tracks = []
            del_tracks = []
            for cluster in self.get_selected_clusters():
                alive_track, del_track = self.get_tracks(cluster.leaves[0])
                tracks.extend(alive_track)
                del_tracks.extend(del_track)
        self.main_view.radarFig.set_track_data(tracks, del_tracks)

    # END UPDATERS


class ComboViewTrack(ComboView):
    def __init__(self, background, meas_model, track_gate):
        self.meas_model = meas_model
        self.track_gate = track_gate
        super().__init__(background)

    def __createRadarFig__(self, ax, background):
        return TrackFig(ax, background, self.meas_model, self.track_gate)
