from scipy.stats import chi2
import numpy as np
import cmath
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
import util
import multivariate
import multivariate as mv
import model.model as model
import cv2

class TrackGate2:
    def __init__(self, gate_probability):
        self.gate_probability = gate_probability
        self.gamma = chi2(df=2).ppf(gate_probability)
        print("Gamma:" + str(self.gamma))

    def is_inside(self, track_node, measurement):
        nu = measurement.value - track_node.z_hat
        return nu.T.dot(np.linalg.solve(track_node.B, nu)) < self.gamma  # nu^T * B^-1 * nu < gamma

    def gate_diff(self, track_node, measurement):
        nu = measurement.value - track_node.z_hat
        return nu.T.dot(np.linalg.solve(track_node.B, nu)) - self.gamma

    def get_2D_gate(self, track_node, num_points=40):
        zx, zy = track_node.z_hat[0], track_node.z_hat[1]
        B_inv = np.linalg.inv(track_node.B)
        a, b, c, d = B_inv[0, 0], B_inv[0, 1], B_inv[1, 1], self.gamma
        r = math.sqrt(c*d/(a*c - b**2))
        x1 = np.linspace(-r, r, num=num_points)
        x2 = np.flip(x1, 0)
        y1_f = np.vectorize(lambda x: cmath.sqrt(((-a*c + b**2)*(x**2) + c*d)/(c**2)).real - b*x/c + zy)
        y2_f = np.vectorize(lambda x: -cmath.sqrt(((-a*c + b**2)*(x**2) + c*d)/(c**2)).real - b*x/c + zy)
        y1 = y1_f(x1)
        y2 = y2_f(x2)
        x1 = np.add(x1, zx)
        x2 = np.add(x2, zx)
        return np.concatenate((x1, x2)), np.concatenate((y1, y2))


def innovate_track_nodes(track_nodes, measurement_model):
    return [t if t.isPosterior else t.innovateTrack(None, measurement_model) for t in track_nodes]


class MeasurementModel:
    def __init__(self, H, R, F, Q, dt, P_D, P_X):
        self.H = H
        self.R = R
        self.F = F
        self.Q = Q
        self.dt = dt
        self.Q_correction = self.Q[0, 0]/self.dt**2
        self.P_D = P_D
        self.P_X = P_X

    def initialize(self, measurement):
        mean = np.array([measurement.value[0], measurement.init_speed[0], measurement.value[1], measurement.init_speed[1]])
        vx_std_sq = measurement.init_speed_var[0] - self.Q_correction
        vy_std_sq = measurement.init_speed_var[1] - self.Q_correction

        covariance = np.diag((self.R[0,0], vx_std_sq, self.R[1,1], vy_std_sq))
        return mean, covariance

    def predict(self, track_node : model.TrackNode):
        track_node.est_prior = self.F.dot(track_node.est_posterior)
        track_node.cov_prior = self.F.dot(track_node.cov_posterior).dot(self.F.T) + self.Q
        track_node.z_hat = self.H.dot(track_node.est_prior)
        track_node.B = self.H.dot(track_node.cov_prior).dot(self.H.T) + self.R
        track_node.isPosterior = False
        track_node.PD = self.P_D
        track_node.PX = self.P_X

    def update(self, track_node, z):
        kalman_gain = track_node.cov_prior.dot(self.H.T).dot(np.linalg.inv(track_node.B))
        est_posterior = track_node.est_prior + np.dot(kalman_gain, z - track_node.z_hat)
        cov_posterior = track_node.cov_prior - np.dot(kalman_gain, np.dot(track_node.B, kalman_gain.T))
        return est_posterior, cov_posterior


class ProbMeasModel(MeasurementModel):
    def __init__(self, H, R, F, Q, dt, PD, PX, land_mask, extent):
        # land_mask:  Area of detectable area.
        super().__init__(H, R, F, Q, dt, PD, PX)
        self.PO_base = 1 - PD - PX

        xlim = (extent[2], extent[3])
        ylim = (-extent[1], -extent[0])

        # Resize land mask:
        land_mask = cv2.resize(land_mask.astype('uint8'), (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) > 0
        self.land_mask = ~np.flip(land_mask, axis=0).T  # (inv(y), x) -> (y, x) -> (x, y)
        nx, ny = self.land_mask.shape

        # Calc position:
        self.dxy_sq = ((xlim[1] - xlim[0])/nx)*((ylim[1] - ylim[0])/ny)
        self.x = np.linspace(xlim[0], xlim[1], nx, endpoint=True).reshape((nx, 1))
        self.y = np.linspace(ylim[0], ylim[1], ny, endpoint=True)

        X, Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((X.T, Y.T))

        print("ProbMeasModel: land-dxy: {}".format(math.sqrt(self.dxy_sq)))

        # Create working arrays:
        self.out = np.empty((nx, ny))
        self.x_copy = np.empty((nx, 1))
        self.y_copy = np.empty((ny,))

    def predict(self, track_node : model.TrackNode):
        super().predict(track_node)
        pdf_values = mv.pdf_bivariate(self.x, self.y,
                                       track_node.z_hat[0], track_node.z_hat[1],
                                       track_node.B[0, 0], track_node.B[1, 1],
                                       self.out, self.x_copy, self.y_copy)

        pdf_values[self.land_mask] = 0

        integral = np.sum(pdf_values)*self.dxy_sq  # [0, 1]
        PD_true = (self.P_D + self.P_X)*integral
        track_node.PD = self.P_D + 2*(integral - 0.5)*self.P_X#max(PD_true, (self.P_D + self.P_X)*0.9)
        track_node.PX = 1 - track_node.PD - self.PO_base
        track_node.i_det = integral
        #print("Track Node: I={:.2f}, {:.2f}:{:.2f}:{:.2f}".
        #      format(integral, PD_true, 1 - track_node.PX - track_node.PD, track_node.PX))
