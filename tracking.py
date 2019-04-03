from scipy.stats import chi2
import numpy as np
import cmath
import math
import multivariate as mv
import model.model as model
import cv2
from model.model import *
from numpy.linalg import inv

MULTI_C_2 = 1/math.sqrt((2*math.pi)**2)
_EYE_4 = np.eye(4, 4)

def safe_inv(A):
    return inv(A)

    #return np.linalg.lstsq(A, _EYE_4)[0]

class TrackGate2:
    def __init__(self, gate_probability):
        self.gate_probability = gate_probability
        self.gamma = chi2(df=2).ppf(gate_probability)

    def is_inside(self, track_node, measurement):
        nu = measurement.value - track_node.z_hat
        return nu.T.dot(np.linalg.solve(track_node.B, nu)) < self.gamma  # nu^T * B^-1 * nu < gamma

    def gate_diff(self, track_node, measurement):
        nu = measurement.value - track_node.z_hat
        return nu.T.dot(np.linalg.solve(track_node.B, nu)) - self.gamma

    def get_2D_gate(self, track_node, num_points=40):
        zx, zy = track_node.z_hat[0], track_node.z_hat[1]
        B_inv = inv(track_node.B)
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

    def __str__(self):
        return "Track Gate: P = {:.2f}. Gamma: {:.2f}".format(self.gate_probability, self.gamma)


def innovate_track_nodes(track_nodes, measurement_model):
    return [t if t.isPosterior else t.innovateTrack(None, measurement_model) for t in track_nodes]


class MeasurementModel:
    __slots__ = 'F', 'Finv', 'Q', 'R', 'H', 'dt', 'v_var', 'del_only_meas', 'P_D', 'P_X', 't44', 't22'


    def __init__(self, q_std_cont, r_std_discrete, dt, P_D, P_X, vmax=5):
        # State matrices:
        self.F = np.identity(4)
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.Finv = inv(self.F)
        self.Q = np.array([[dt ** 4 / 4, dt ** 3 / 2, 0, 0],
                      [dt ** 3 / 2, dt ** 2, 0, 0],
                      [0, 0, dt ** 4 / 4, dt ** 3 / 2],
                      [0, 0, dt ** 3 / 2, dt ** 2]])* (q_std_cont ** 2)
        self.R = np.identity(2) * (r_std_discrete**2)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Other
        self.dt = dt
        self.v_var = (vmax/3)**2

        # Detection/Deletion
        self.del_only_meas = False
        self.P_D = P_D
        self.P_X = P_X

        # TEMP arrays.
        self.t44 = np.ascontiguousarray(np.array((4, 4), dtype=np.float64))
        self.t22 = np.ascontiguousarray(np.array((2, 2), dtype=np.float64))

    def initialize(self, detection: Detection):
        mean = np.array((detection.pos[0], 0,
                         detection.pos[1], 0))
        vx_std_sq = self.v_var
        vy_std_sq = self.v_var
        covariance = np.diag((self.R[0,0], vx_std_sq, self.R[1,1], vy_std_sq))
        return mean, covariance

    def predict(self, track_node: model.TrackNode):
        track_node.est_prior = self.F.dot(track_node.est_posterior)
        track_node.cov_prior = ((self.F @ track_node.cov_posterior) @ self.F.T) + self.Q
        #track_node.cov_prior = np.dot(np.dot(self.F, track_node.cov_posterior, out=self.t44), self.F.T, out=self.t44) + self.Q
        track_node.z_hat = self.H.dot(track_node.est_prior)
        track_node.B = ((self.H @ track_node.cov_prior) @ self.H.T) + self.R
        #track_node.B = np.dot(np.dot(self.H, track_node.cov_prior, out=self.t22), self.H.T, out=self.t22) + self.R
        track_node.Binv = inv(track_node.B)
        track_node.mg = MULTI_C_2/math.sqrt(track_node.B[0, 0]*track_node.B[1, 1])
        track_node.isPosterior = False
        if self.del_only_meas and track_node.measurement is None:
            track_node.PD = self.P_D
            track_node.PX = 0
        else:
            track_node.PD = self.P_D
            track_node.PX = self.P_X

    def update(self, track_node, z):
        kalman_gain = track_node.cov_prior.dot(self.H.T).dot(track_node.Binv)
        est_posterior = track_node.est_prior + np.dot(kalman_gain, z - track_node.z_hat)
        cov_posterior = track_node.cov_prior - np.dot(kalman_gain, np.dot(track_node.B, kalman_gain.T))
        return est_posterior, cov_posterior

    def RTS(self, track: List[TrackNode]):
        iterator = reversed(track)
        last_node = next(iterator)
        new_last_node = TrackNode(last_node.scan_idx, None, last_node.est_posterior,
                                  last_node.cov_posterior, last_node.target, last_node.measurement)
        new_nodes = [new_last_node]
        for node in iterator:
            prev_node = new_nodes[-1]
            try:
                invCovPrior = inv(node.cov_prior)
            except np.linalg.linalg.LinAlgError:
                # print("ERROR: {}, {}".format(node.measurement, prev_node.measurement))
                lstinv = np.linalg.lstsq(node.cov_prior, _EYE_4)
                # for A in lstinv:
                #     print("A: {}".format(A))
                invCovPrior = lstinv[0]

            Ck = node.cov_posterior.dot(self.F.T).dot(invCovPrior)
            new_est = node.est_posterior + Ck.dot(prev_node.est_posterior - node.est_prior)
            new_cov = node.est_posterior + Ck.dot(prev_node.cov_posterior - node.cov_prior).dot(Ck.T)
            new_node = TrackNode(node.scan_idx, None, new_est, new_cov, node.target, node.measurement)
            new_nodes[-1].parent = new_node
            new_nodes.append(new_node)
        return list(reversed(new_nodes))


class ProbMeasModel(MeasurementModel):
    def __init__(self, q_std_cont, r_std_discrete, dt, PD, PX, land_mask, extent):
        # land_mask:  Area of detectable area.
        super().__init__(q_std_cont, r_std_discrete, dt, PD, PX)
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

        print("ProbMeasModel: land-dxy: {:.3f}".format(math.sqrt(self.dxy_sq)))

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
        PX_adjusted = self.P_X*(1-integral)
        track_node.PD = self.P_D/(1-self.P_X)*(1-PX_adjusted)
        track_node.PX = PX_adjusted
        track_node.i_det = integral
        #print("Track Node: I={:.2f}, {:.2f}:{:.2f}:{:.2f}".
        #      format(integral, PD_true, 1 - track_node.PX - track_node.PD, track_node.PX))
