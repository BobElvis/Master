import numpy as np


class ConstantVelocityModel:
    """Constant velocity motion model."""

    def __init__(self, q):
        self.q = q

    def getFQ(self, dT):
        F = np.array([[1, 0, dT, 0],
                      [0, 1, 0, dT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.array([[dT ** 3 / 3, 0, dT ** 2 / 2, 0],
                      [0, dT ** 3 / 3, 0, dT ** 2 / 2],
                      [dT ** 2 / 2, 0, dT, 0],
                      [0, dT ** 2 / 2, 0, dT]]) * self.q
        return F, Q

    def step(self, xprev, Pprev, dT):
        F, Q = self.getFQ(dT)
        x = F * xprev
        P = F * Pprev * F.T + Q
        return x, P


def position_measurement(x):
    """Velocity measurement model."""
    H = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return H * x, H


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.array([[0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return H * x, H