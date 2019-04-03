from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import matplotlib
matplotlib.use("QT5AGG")
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.figures import *
from timer import *
from dataloader import BaseLoader
tim = SimpleTimer()


class Callable:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        pass


class DynApp(QWidget):
    def __init__(self, dataloader: BaseLoader, background, i=0, show_camera = True):
        super().__init__()
        self.dataloader = dataloader
        self.idx_edit_width = 125
        self.cam_edit_width = 100
        self.lag = 0

        # State:
        self.draw_pending = False
        self.i = i

        # UI elements
        self.layoutWrapper = QVBoxLayout() # Contains toolbar and content
        self.layoutToolbar = QHBoxLayout()
        self.layoutContent = QHBoxLayout()

        self.main_view = self.__createMainView__(background)
        self.prevBtn = QPushButton("Prev")
        self.nextBtn = QPushButton("Next")
        self.prevMeasBtn = QPushButton("Prev Meas")
        self.nextMeasBtn = QPushButton("Next Meas")
        self.timeLabel = QLabel("----")
        self.idxEdit = QLineEdit()
        self.lagEdit = QLineEdit()

        # Camera:
        self.camShowBox = QCheckBox("Cam")
        self.camShowBox.setChecked(show_camera)
        self.improveCamBox = QCheckBox("Cam+")
        self.improveCamBox.setChecked(False)

        # To handle figure size:
        self.fig_size_init = self.main_view.figure.get_size_inches()
        self.fig_size_inited = False

        # Range:
        self.edit_lim = [QLineEdit() for _ in range(4)]
        l = self.main_view.radarFig.get_lims()
        for k, edit in enumerate(self.edit_lim):
            edit.setText(str((l[0] + l[1])[k]))

        # Setup
        self.initUI()
        self.change_idx(self.i)
        self.setListeners()

        # Get last screen config:
        self.settings = QSettings('App', 'myApp')
        self.resize(self.settings.value("size", QSize(270, 225)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
        self.show()

    def connect(self, event, handler):
        event.connect(handler)
        event.connect(self.on_event)

    def on_event(self):
        if self.draw_pending:
            self.main_view.update_view(True, True)
            self.draw_pending = False

    def __createMainView__(self, background):
        return ComboView(background)

    def initUI(self):
        self.layoutToolbar.setAlignment(Qt.AlignCenter)
        self.layoutToolbar.addWidget(self.camShowBox)
        self.layoutToolbar.addWidget(self.improveCamBox)
        self.layoutToolbar.addWidget(self.lagEdit)
        self.layoutToolbar.addWidget(self.prevBtn)
        self.layoutToolbar.addWidget(self.nextBtn)
        self.layoutToolbar.addWidget(self.prevMeasBtn)
        self.layoutToolbar.addWidget(self.nextMeasBtn)
        for edit in self.edit_lim:
            self.layoutToolbar.addWidget(edit)
            edit.setFixedWidth(self.idx_edit_width*0.75)
        self.layoutToolbar.addWidget(self.timeLabel)
        self.layoutToolbar.addWidget(self.idxEdit)

        self.timeLabel.setFixedWidth(self.idx_edit_width*2)
        self.idxEdit.setFixedWidth(self.idx_edit_width)
        self.layoutWrapper.addLayout(self.layoutToolbar)
        self.layoutWrapper.addLayout(self.layoutContent)
        self.layoutContent.addWidget(self.main_view)
        self.camShowBox.setFixedWidth(self.cam_edit_width)
        self.setLayout(self.layoutWrapper)
        self.setWindowTitle("Surveillance")

    def setListeners(self):
        self.connect(self.lagEdit.editingFinished, self.onLagEdit)
        self.connect(self.prevBtn.clicked, self.prev)
        self.connect(self.nextBtn.clicked, self.next)
        self.connect(self.prevMeasBtn.clicked, self.prevMeas)
        self.connect(self.nextMeasBtn.clicked, self.nextMeas)
        self.connect(self.idxEdit.editingFinished, self.onIdxEdit)

        self.camShowBox.stateChanged.connect(self.onShowCamChanged)
        self.improveCamBox.stateChanged.connect(self.onImproveCamChanged)

        for edit in self.edit_lim:
            edit.editingFinished.connect(self.on_lim_changed)

    def resizeEvent(self, event):
        self.on_fig_size_changed()
        return super().resizeEvent(event)

    def on_initialized(self):
        print("INITIALIZED")
        self.main_view.radarFig.on_resize()
        self.main_view.camFig.on_resize()
        self.main_view.update_margins()
        self.main_view.update_view(force_full=True)

    def on_fig_size_changed(self):
        if self.fig_size_inited:
            self.main_view.update_margins()
            self.main_view.camFig.on_resize()
            self.main_view.radarFig.on_resize()

    def event(self, event):
        if not self.fig_size_inited:
            s1 = self.main_view.figure.get_size_inches()
            s2 = self.fig_size_init
            if s1[0] != s2[0] and s1[1] != s2[1]:
                self.fig_size_inited = True
                self.on_initialized()
        return super().event(event)

    def change_idx(self, i):
        prev_i = self.i
        self.i = min(i, len(self.dataloader)-1)
        self.on_idx_changed(self.i, prev_i)

    def on_idx_changed(self, curr_i, prev_i):
        scan = self.dataloader[self.i]

        # LAG:
        lag_idx = max(0, curr_i - self.lag)
        lag_image = scan.radar_img
        for i in range(lag_idx, curr_i):
            lag_image += self.dataloader[i].radar_img
        scan.radar_img = lag_image

        self.main_view.set_scan(scan)
        self.timeLabel.setText(str(scan.time))
        self.idxEdit.setText(str(curr_i))
        self.draw()

    def on_lim_changed(self):
        limits = [float(edit.text()) for edit in self.edit_lim]
        self.main_view.radarFig.set_lims(limits[:2], limits[2:])
        self.main_view.update_view(force_full=True)

    def onShowCamChanged(self):
        self.main_view.set_show_camera(self.camShowBox.isChecked())
        self.main_view.on_show_camera_changed()

    def onImproveCamChanged(self):
        self.main_view.camFig.improve_camera = self.improveCamBox.isChecked()
        self.main_view.update_view(radar=False)

    def prev(self):
        self.change_idx((self.i - 1) % len(self.dataloader))

    def next(self):
        self.change_idx((self.i + 1) % len(self.dataloader))

    def prevMeas(self):
        self.nextPrevMeas(-1)

    def nextMeas(self):
        self.nextPrevMeas(1)

    def nextPrevMeas(self, di):
        i = self.i
        while True:
            i = (i + di) % len(self.dataloader)
            item = self.dataloader[i]
            if len(item) > 0:
                break
        self.change_idx(i)

    def onIdxEdit(self):
        self.change_idx(int(float(self.idxEdit.text())))

    def onLagEdit(self):
        self.lag = int(float(self.lagEdit.text()))
        self.on_idx_changed(self.i, self.i)

    def closeEvent(self, e):
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        e.accept()

    def draw(self, radar=None, now=False):
        # Radar: None -> Both. True -> Only radar. False -> Only camera.
        self.draw_pending = True
        if now:
            self.main_view.update_view(True, True)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Right:
            self.next()
            self.on_event()
        elif key == Qt.Key_Left:
            self.prev()
            self.on_event()
        elif key == Qt.Key_Up:
            self.nextMeas()
            self.on_event()
        elif key == Qt.Key_Down:
            self.prevMeas()
            self.on_event()


class ComboView(FigureCanvas):
    def __init__(self, background):
        self.first = True
        self.m1 = (0.05, 0.0, 0.45, 0.75, 0.05)  # h_space, top, bottom, left, right
        self.m2 = (0.0, 0.1, self.m1[2], self.m1[3], 0.1)
        dpi = 180
        self.figsize = (1, 1)
        fig = Figure(figsize=self.figsize, dpi=dpi)
        super().__init__(fig)

        self.gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 2.82])
        self.gs2 = gridspec.GridSpec(1, 1, height_ratios=[1])
        self.camFig = CameraFig(self.figure.add_subplot(self.gs1[0]))
        self.radarFig = self.__createRadarFig__(self.figure.add_subplot(self.gs1[1]), background)

        # Set sizes
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.pos_radar = None
        self.margins = None
        self.set_show_camera(True)

    def set_show_camera(self, show_camera):
        print("Setting show camera")
        self.camFig.ax.set_visible(show_camera)
        self.pos_radar = self.gs1[1] if show_camera else self.gs2[0]
        self.margins = self.m1 if show_camera else self.m2

    def on_show_camera_changed(self):
        self.update_margins()
        self.camFig.on_resize()
        self.radarFig.on_resize()
        self.update_view(force_full=True)

    def update_margins(self):
        figsize = self.figure.get_size_inches()
        abs_margins = {
            "hspace": self.margins[0]/ figsize[1],
            "top": 1 - self.margins[1]/ figsize[1],
            "bottom": self.margins[2]/ figsize[1],
            "left": self.margins[3]/ figsize[0],
            "right": 1 - self.margins[4]/ figsize[0],
        }
        self.figure.subplots_adjust(**abs_margins)
        self.radarFig.ax.set_position(self.pos_radar.get_position(self.figure))

    def set_scan(self, scan):
        self.camFig.set_data(scan)
        self.radarFig.set_data(scan)

    def update_view(self, radar=True, camera=True, force_full = False):
        if force_full or self.first:
           self.draw()
           self.first = False
        else:
            if camera and self.camFig.ax.get_visible():
                self.camFig.draw_artists()
            if radar:
                self.radarFig.draw_artists()
            self.figure.canvas.update()
            self.figure.canvas.flush_events()

    def __createRadarFig__(self, ax, background):
        return RadarFig(ax, background)