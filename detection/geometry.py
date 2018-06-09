import math

import cv2
import numpy as np
from numpy.linalg import norm


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def distance_to_line_np(s1, s2, p):
    # s1 = (1,2), s2 = (1,2), p = (n, 2)
    c = np.abs(np.cross(s2 - s1, p - s1))
    n = norm(s2-s1)
    return c / n


def distance_to_line(p1, p2, s):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = abs(y_diff * s[0] - x_diff * s[1] + p2[0] * p1[1] - p2[1] * p1[0])
    den = math.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / den


def contour_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy