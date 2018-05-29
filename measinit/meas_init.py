import numpy as np
import math
import time
from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


class MeasInitBase:
    def __init__(self, area, targets_per_scan, vmax):
        self.target_density = targets_per_scan/area
        self.variance = (vmax/3)**2

    def init_measurements(self, measurements):
        for m in measurements:
            m.init_speed = np.array([0, 0])
            m.init_speed_var = np.array([self.variance, self.variance])
            m.density = self.target_density


class MeasInit:
    def __init__(self, boundary, dt, P_D, num_detections, targets_per_scan, vmax, init_speed=True):
        # boundary nx2 np array of vertices
        # offsets nx1 np array of offsets
        # speed nx1 np array of speed out of each wall.
        # weight nx1 np array of probability that a target appears from each wall

        self.B = boundary
        self.detections = np.arange(0, num_detections) #0, 1, num_det-1;
        self.dt = dt
        self.PD = P_D
        self.vmax = vmax
        self.init_speed = init_speed

        # Calculating some intermediate arrays:
        self.det_divisor = (self.detections + 1)*dt
        self.det_factor = np.power(1-P_D, self.detections)

        self.velocities = np.zeros((2, self.B.n))
        self.probabilities = np.zeros(self.B.n)

        # This constant can be found by using self.calcIntegral(dxy)
        c_pre = 8.522e-05  # dxy=1.00
        self.c = targets_per_scan * c_pre

    def init_measurements(self, measurements):
        import time
        t = time.time()
        for m in measurements:
            self.init_single_meas(m.value)
            prob_tot = np.max(self.probabilities)
            if self.init_speed:
                rate_wall = self.probabilities*self.B.weight
                vel_weight = rate_wall/sum(rate_wall)
                v_tot = np.sum(self.velocities*vel_weight, axis=1)  # Along cols.
            else:
                v_tot = np.array([0, 0])
            m.init_speed = v_tot
            m.init_speed_var = ((self.vmax - np.abs(v_tot))/3)**2
            m.density = prob_tot*self.c
            #print("M Init: ({:.2f}, {:.2f}), den={:.2e}, p={:.4f}, speed=({:.2f}, {:.2f})"
            #      .format(m.value[0], m.value[1], m.density, prob_tot, m.init_speed[0], m.init_speed[1]))
        print("Meas_init: Time used: {:.5f}".format(time.time()-t))

    def init_single_meas(self, point):
        for j in range(self.B.n):
            if self.B.speed[j] == 0:
                continue
            v, p = self.init_meas_wall(point, j)
            self.velocities[:, j] = v
            self.probabilities[j] = p

    def init_meas_wall(self, point, idx):
        length = self.B.length[idx]
        offset = self.B.offset[idx]
        v = self.B.speed[idx]

        # Transform to new coordinate system:
        qa = np.dot(self.B.rot_matrix[idx], point - self.B.p1[idx, :])
        x, y = qa[0], qa[1]

        # Calculate a_factor:
        d_left = x
        d_right = length - x
        a_left = math.atan2(y, d_left) % (2*np.pi)
        a_right = math.atan2(y, d_right) % (2*np.pi)
        #print(math.atan2(y, d_left) % (2*np.pi))
        a_factor_left = self.smooth_angle(a_left, self.B.angle_start_prev[idx], self.B.angle_stop_prev[idx])
        a_factor_right = self.smooth_angle(a_right, self.B.angle_start_next[idx], self.B.angle_stop_next[idx])
        a_factor = a_factor_left*a_factor_right

        # Calculate speed based on distance from edge:
        if x < 0:
            dx = x
        elif x > length:
            dx = x - length
        else:
            dx = 0
        d = math.sqrt(dx**2 + y**2)*math.cos(offset)
        p = self.calcProb(d, v, v/3)*a_factor

        # Transform back to coordinate system:
        beta = math.atan2(y, dx)
        angle = self.B.phi[idx] - offset + beta
        vx = v*math.cos(angle)
        vy = v*math.sin(angle)
        return np.array([vx, vy]), p

    def calcProb(self, d, speed, sigma):
        x = (1-norm.cdf(d/self.det_divisor, loc=speed, scale=sigma))*self.det_factor
        return np.sum(x)*self.PD

    @staticmethod
    def smooth_angle(phi, a_start, a_stop):
        if phi >= a_stop:
            return 0
        elif phi <= a_start:
            return 1
        else:
            return 1 - (phi - a_start)/(a_stop - a_start)

    def calcIntegral(self, dxy, show=False):
        B = self.B.p1
        x = np.arange(np.min(B[:, 0]), np.max(B[:, 0])+dxy, dxy)
        y = np.arange(np.min(B[:, 1]), np.max(B[:, 1])+dxy, dxy)
        values = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                if not self.B.is_point_inside(x[i], y[j]):
                    continue
                self.init_single_meas(np.array((x[i], y[j])))
                values[i, j] = np.max(self.probabilities)
        if show:
            plt.imshow(values, cmap='hot', interpolation='nearest')
            plt.show()
        return 1/(np.nansum(values)*dxy*dxy)
