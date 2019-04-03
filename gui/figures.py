import numpy as np
from util import readImg, readMask
import cv2
import gui.background as bg
from gui.image_draw import BackgroundDrawer

import timer
import util
from matplotlib.axes import Axes
t = timer.SimpleTimer()


class MaskImgView:
    def __init__(self, ax, mask, color, extent):
        self.a = color[3]
        self.img = bg.mask2img(mask, color)
        if not isinstance(extent, tuple):
            extent = (-extent, extent, -extent, extent)
        self.h = ax.imshow(self.img, extent=extent, interpolation='nearest')

    def set_data(self, mask):
        self.img[:, :, 3] = np.array(mask, dtype=np.float)*self.a
        self.h.set_data(self.img)

    def draw(self, cached_renderer):
        self.h.update_view(cached_renderer)


class CameraFig(object):
    def __init__(self, ax):
        self.ax = ax
        self.ax.axis('off')
        self.img_artist = None
        self.data = None
        self.improve_camera = False

    def set_data(self, scan):
        path = scan.camera_path
        data = None if path is None else util.readImg(path)

        if data is None:
            data = np.zeros((1,1))
        elif self.improve_camera:
            data = self.improve_camera_func(data)
        self.data = data

        if self.img_artist is None:
            self.img_artist = self.ax.imshow(self.data)
        else:
            self.img_artist.set_data(self.data)

    @staticmethod
    def improve_camera_func(img):
        # Gamma correction:
        gamma = 2
        img = np.power(img/255, 1/gamma)*255
        img = img.astype('uint8')
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lum, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        cl = clahe.apply(lum)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

    def on_resize(self):
        pass

    def draw_artists(self):
        self.ax.draw_artist(self.img_artist)


class RadarFig(object):
    def __init__(self, ax: Axes, background):
        self.ax = ax
        self.range = background.radar_range
        bg, ext = background.get_img()
        self.hbg = self.ax.imshow(bg, extent=ext, interpolation='none')
        self.bg_wrapper = BackgroundDrawer(self.hbg)
        self.hRadarData = None
        self.hMeas = None
        self.ax.set_xlabel('East')
        self.ax.set_ylabel('North')
        self.set_lims(background.x_lim, background.y_lim)
        self.ax.title.set_visible(False)

        self.show_meas_idx = False
        self.h_text_meas = [] if self.show_meas_idx else None
        self.h_text_meas_n = 0

    def on_resize(self):
        self.bg_wrapper.reset()

    def set_lims(self, x_lim, y_lim):
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.bg_wrapper.reset()

    def get_lims(self):
        return self.ax.get_xlim(), self.ax.get_ylim()

    def set_data(self, scan):
        radar_img = scan.radar_img
        #radar_img = cv2.resize(radar_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        mx = scan.m[:, 0]
        my = scan.m[:, 1]

        if self.hRadarData is None:
            if radar_img is not None:
                extent = (-self.range, self.range, -self.range, self.range)
                if scan.radar_img.dtype == 'bool':
                    self.hRadarData = MaskImgView(self.ax, radar_img, color=(0, 1, 1, 0.5), extent=extent)
                else:
                    self.hRadarData = self.ax.imshow(radar_img, extent=extent, interpolation='nearest')
            self.hMeas = self.ax.plot(mx, my, 'r+', linestyle="None")[0]
        else:
            self.hRadarData.set_data(radar_img)
            self.hMeas.set_xdata(mx)
            self.hMeas.set_ydata(my)

        # Text:
        if self.show_meas_idx:
            for i in range(len(scan)):
                x, y = mx[i], my[i]
                if i >= len(self.h_text_meas):
                    self.h_text_meas.append(self.ax.text(x, y, str(i+1), color='white', fontsize=15))
                else:
                    #self.h_text_meas[i] = self.ax.text(x, y, str(i), color='white', fontsize=15)
                    self.h_text_meas[i].set_position((x, y))
            self.h_text_meas_n = len(scan)

    def draw_artists(self):
        self.bg_wrapper.draw(self.ax._cachedRenderer)
        if self.hRadarData is not None:
            self.hRadarData.draw(self.ax._cachedRenderer)
        self.ax.draw_artist(self.hMeas)

        for i in range(self.h_text_meas_n):
            self.ax.draw_artist(self.h_text_meas[i])

    def get_aspect(self):
        x_lim, y_lim = self.get_lims()
        return (y_lim[1] - y_lim[0])/(x_lim[1] - x_lim[0])  # height/width


class TrackFig(RadarFig):
    def __init__(self, ax, background, meas_model, track_gate):
        super().__init__(ax, background)
        self.meas_model = meas_model
        self.track_gate = track_gate
        self.h_track = []
        self.tracks = []
        self.conf_split = 0

        # Design:
        self.marker_size = 15

        self.gate_one_behind = True

        #print(self.get_aspect())

    def set_track_data(self, tracks, del_tracks=None):
        del_tracks = [] if del_tracks is None else del_tracks
        self.tracks = [(t, True) for t in tracks] + [(t, False) for t in del_tracks]
        self.tracks = sorted(self.tracks, key=lambda t: t[0][0].scan_idx, reverse=False)

        for idx, tc in enumerate(self.tracks):
            track, alive = tc  # track is a list of track_nodes
            n_nodes = len(track)
            pos, missed = np.zeros((n_nodes, 2)), np.full((n_nodes,), True)
            for j, track_node in enumerate(track):
                pos[j, :] = self.meas_model.H.dot(track_node.est_posterior)
                missed[j] = track_node.measurement is None

            if alive and self.track_gate is not None:
                if self.gate_one_behind and len(track) > 1:
                    node = track[-2]
                else:
                    node = track[-1]

                if node is not None:
                    if node.i_det is not None:
                        print("I: {:.3f}, {:.3f}:{:.3f}:{:.3f}".
                            format(node.i_det, node.PD, 1-node.PD-node.PX, node.PX))
                    gate_x, gate_y = self.track_gate.get_2D_gate(node)
                    z_hat = np.reshape(node.z_hat, (1, 2))
            else:
                gate_x, gate_y = [], []
                z_hat = np.zeros((0, 2))

            if alive:
                plot_style = '-'
            else:
                plot_style = '--'

            m = pos[np.logical_not(missed), :]
            n = pos[missed, :]

            if idx >= len(self.h_track):
                h_path = self.ax.plot(pos[:, 0], pos[:, 1], plot_style)[0]
                color = h_path.get_color()
                h_m = self.ax.scatter(m[:, 0], m[:, 1], marker='o', color=color, s=self.marker_size)
                h_n = self.ax.scatter(n[:, 0], n[:, 1], marker='x', color=color, s=self.marker_size)
                h_p = self.ax.scatter(z_hat[:, 0], z_hat[:, 1], marker='*', color=color, s=self.marker_size)
                h_gate = self.ax.plot(gate_x, gate_y, color=color)[0]
                self.h_track.append((h_path, h_gate, h_m, h_n, h_p))
            else:
                self.h_track[idx][0].set_ls(plot_style)
                self.h_track[idx][0].set_xdata(pos[:, 0])
                self.h_track[idx][0].set_ydata(pos[:, 1])
                self.h_track[idx][1].set_xdata(gate_x)
                self.h_track[idx][1].set_ydata(gate_y)
                self.h_track[idx][2].set_offsets(m)  # Set position of scatter.
                self.h_track[idx][3].set_offsets(n)  # Set position of scatter.
                self.h_track[idx][4].set_offsets(z_hat)  # Set position of scatter.

    def draw_artists(self):
        super().draw_artists()
        for i in range(0, len(self.tracks)):
            self.ax.draw_artist(self.h_track[i][0])
            self.ax.draw_artist(self.h_track[i][1])
            self.ax.draw_artist(self.h_track[i][2])
            self.ax.draw_artist(self.h_track[i][3])
            self.ax.draw_artist(self.h_track[i][4])
        self.ax.draw_artist(self.hMeas)
