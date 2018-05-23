import numpy as np
import multivariate as mv
import time
import math

class ProbBase:
    def meas_prob(self, m_value, z_hat, B):
        return mv.multi_gaussian(m_value - z_hat, B)


class ProbArea(ProbBase):
    __slots__ = 'mask_impassable_neg', 'dxy_sq', 'x', 'y', 'pos', 'out'

    def __init__(self, mask_impassable, range):
        nx = mask_impassable.shape[0]
        ny = mask_impassable.shape[1]

        self.mask_impassable_neg = ~mask_impassable#.reshape(n ** 2)

        self.dxy_sq = (range*2/nx)*(range*2/ny)
        self.x = np.linspace(-range, range, nx)#.reshape((nx, 1))
        self.y = np.linspace(-range, range, ny).reshape((ny, 1))
        self.out = np.empty((nx, ny))

        self.x_copy = np.empty(nx)
        self.y_copy = np.empty((ny, 1))

        X, Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((X, Y))

    def meas_prob(self, m_value, z_hat, B):
        t1 = time.time()
        pdf_values = mv.pdf_bivariate2(self.x, self.y, z_hat[0], z_hat[1], B[0,0], B[1,1],
                                       self.out, self.x_copy, self.y_copy)
        i_tot = np.sum(pdf_values)*self.dxy_sq

        #im = np.ma.masked_array(pdf_values, self.mask_impassable_neg); i_unpassable = im.sum()*self.dxy_sq
        pdf_values[self.mask_impassable_neg] = 0
        i_unpassable = np.sum(pdf_values)*self.dxy_sq
        #i_unpassable = np.sum(np.multiply(pdf_values, self.mu_mask, out=pdf_values))*self.dxy_sq

        t1 = time.time() - t1

        i_passable = 1 - (i_unpassable)
        r = 1/i_passable
        p = super().meas_prob(m_value, z_hat, B)*r

        print("Time: {:.5f}, p={:.2f}->{:.2f} I = {:.3f}/{:.3f}, R={:.4f}, Var=({:.1f})".
              format(t1, p/r*100, p*100, i_tot, i_unpassable, r, math.sqrt(B[0,0]*B[1,1])))
        return p

    def __repr__(self):
        return "dxy = {}".format(math.sqrt(self.dxy_sq))