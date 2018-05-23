import numpy as np
import scipy.stats._multivariate as mv
import scipy as sp
import math

_LOG_2PI = mv._LOG_2PI

def multi_gaussian(x, P):
    # x is a N dimensional vector
    return math.exp(-0.5*x.dot(np.linalg.solve(P, x.T)))/math.sqrt(np.linalg.det(P)*(2*math.pi)**x.shape[0])

def log_pdf(x, mean, prec_U, log_det_cov, rank, out=None, x_copy=None):
    x = np.subtract(x, mean, out=x_copy)  # dev
    np.dot(x, prec_U, out=x)
    np.square(x, out=x)
    y = np.sum(x, axis=-1, out=out)  # maha
    np.add(y, (rank * mv._LOG_2PI + log_det_cov), out=y)
    np.multiply(y, -0.5, out=y)
    return y


def pdf(x, mean, cov, out=None, x_copy=None):
    psd = mv._PSD(cov, allow_singular=False)
    y = log_pdf(x, mean, psd.U, psd.log_pdet, psd.rank, out, x_copy)
    return np.exp(y, out=y)


class Multivariate:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.psd = mv._PSD(cov, allow_singular=False)

    def pdf(self, x, out=None, x_copy=None):
        y = log_pdf(x, self.mean, self.psd.U, self.psd.log_pdet, self.psd.rank, out, x_copy)
        return np.exp(y, out=y)


def pdf_bivariate(x, y, mx, my, var_x, var_y, out=None, x_copy=None, y_copy=None):
    x = np.subtract(x, mx, out=x_copy)
    np.square(x, out=x)
    np.multiply(x, -1 / (2 * var_x), out=x)

    y = np.subtract(y, my, out=y_copy)
    np.square(y, out=y)
    np.multiply(y, -1 / (2 * var_y), out=y)

    #pos = np.add(x, y, out=out)
    pos = np.add(x, y, out=out)
    np.exp(pos, out=pos)
    np.multiply(pos, 1 / (2 * np.pi * math.sqrt(var_x * var_y)), out=pos)
    return pos


def pdf_bivariate2(x, y, mx, my, var_x, var_y, out=None, x_copy=None, y_copy=None):
    x = np.subtract(x, mx, out=x_copy)
    np.square(x, out=x)
    np.multiply(x, -1 / (2 * var_x), out=x)
    np.exp(x, out=x)

    y = np.subtract(y, my, out=y_copy)
    np.square(y, out=y)
    np.multiply(y, -1 / (2 * var_y), out=y)
    np.exp(y, out=y)

    np.multiply(x, 1 / (2 * np.pi * math.sqrt(var_x * var_y)), out=x)
    pos = np.multiply(x, y, out=out)
    #print("x: {} y: {}, Expected out: {}, reality: {}".format(x.shape, y.shape, out.shape, pos.shape))
    return pos
