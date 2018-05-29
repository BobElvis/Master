import numpy as np
from util import readImg, readMask
import cv2
import gui.background as bg

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
        self.h.draw(cached_renderer)


class CameraFig(object):
    def __init__(self, ax):
        self.ax = ax
        self.ax.axis('off')
        self.img_artist = None
        self.data = None

    def set_data(self, scan):
        self.data = scan.camera_img
        if self.data is not None:
            if self.img_artist is None:
                self.img_artist = self.ax.imshow(self.data)
            else:
                self.img_artist.set_data(self.data)

    def draw_artists(self):
        if self.img_artist is not None:
            self.ax.draw_artist(self.img_artist)


class RadarFig(object):
    def __init__(self, ax, background):
        self.ax = ax
        self.range = background.radar_range
        bg, ext = background.get_img()
        self.hbg = self.ax.imshow(bg, extent=ext, interpolation='nearest')
        self.hRadarData = None
        self.hMeas = None
        self.ax.set_xlabel('East')
        self.ax.set_ylabel('North')
        self.set_lims(background.x_lim, background.y_lim)
        self.ax.title.set_visible(False)

    def set_lims(self, x_lim, y_lim):
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)

    def get_lims(self):
        return self.ax.get_xlim(), self.ax.get_ylim()

    def set_data(self, scan):
        if self.hRadarData is None:
            if scan.radar_img is not None:
                extent = (-self.range, self.range, -self.range, self.range)
                if scan.radar_img.dtype == 'bool':
                    self.hRadarData = MaskImgView(self.ax, scan.radar_img, color=(0, 1, 1, 0.5), extent=extent)
                else:
                    self.hRadarData = self.ax.imshow(scan.radar_img, extent=extent, interpolation='nearest')
            self.hMeas = self.ax.plot(scan.mx, scan.my, 'r+', linestyle="None")[0]
        else:
            self.hRadarData.set_data(scan.radar_img)
            self.hMeas.set_xdata(scan.mx)
            self.hMeas.set_ydata(scan.my)

    def draw_artists(self):
        self.ax.draw_artist(self.hbg)
        if self.hRadarData is not None:
            self.hRadarData.draw(self.ax._cachedRenderer)
        self.ax.draw_artist(self.hMeas)

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

        #print(self.get_aspect())

    def set_track_data(self, tracks, conf_tracks=None):
        conf_tracks = [] if conf_tracks is None else conf_tracks
        self.tracks = [(t, True) for t in tracks] + [(t, False) for t in conf_tracks]
        self.tracks = sorted(self.tracks, key=lambda t: t[0][0].target.idx, reverse=False)

        for idx, tc in enumerate(self.tracks):
            track, alive = tc  # track is a list of track_nodes
            n_nodes = len(track)
            pos, missed = np.zeros((n_nodes, 2)), np.full((n_nodes,), True)
            for j, track_node in enumerate(track):
                pos[j, :] = self.meas_model.H.dot(track_node.est_posterior)
                missed[j] = track_node.measurement is None

            if alive:
                node = track[-1]
                if node.i_det is not None:
                    print("I: {:.3f}, {:.3f}:{:.3f}:{:.3f}".
                        format(node.i_det, node.PD, 1-node.PD-node.PX, node.PX))
                plot_style = '-'
                gate_x, gate_y = self.track_gate.get_2D_gate(track[-1])
                z_hat = np.reshape(track[-1].z_hat, (1, 2))
            else:
                plot_style = '--'
                gate_x, gate_y = [], []
                z_hat = np.zeros((0, 2))

            m = pos[np.logical_not(missed), :]
            n = pos[missed, :]

            if idx >= len(self.h_track):
                h_path = self.ax.plot(pos[:, 0], pos[:, 1], plot_style)[0]
                color = h_path.get_color()
                h_m = self.ax.scatter(m[:, 0], m[:, 1], marker='o', color=color)
                h_n = self.ax.scatter(n[:, 0], n[:, 1], marker='x', color=color)
                h_p = self.ax.scatter(z_hat[:, 0], z_hat[:, 1], marker='*', color=color)
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
