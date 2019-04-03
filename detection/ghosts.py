import numpy as np
import detection.split
import cv2
from detection.geometry import contour_centroid

RAD_EPS = 1*np.pi/180


def mark_multiple(dd, positions, cnts_raw, area, area_min, range_ratio):
    n = positions.shape[0]

    if n < 2 or max(area) < area_min:
        return

    data, dist = _get_multiple_data(dd, positions, cnts_raw, area)
    is_multiple = _mark_multiple(dist, data, area_min, range_ratio)
    return is_multiple


def _get_multiple_data(dd, positions, cnts_raw, area):
    dist = np.linalg.norm(positions, axis=1)
    cnts = [dd.scale_measurements(cnt.reshape((-1, 2))) for cnt in cnts_raw]
    angles_cnts = [np.arctan2(cnt[:, 1], cnt[:, 0]) for cnt in cnts]  # List of list of angles
    data = np.array(
        [(np.amin(angles), np.amax(angles), d, area) for angles, d, area in zip(angles_cnts, dist, area)])  # nx2 array
    return data, dist


def _mark_multiple(dists, data, area_min, range_ratio):
    n = dists.shape[0]

    angles_min = data[:, 0]
    angles_max = data[:, 1]
    dists = data[:, 2]

    source = data[:, 3] >= area_min
    delete = None
    deleted_all = np.full((n,), False)

    for i in range(n):
        if not source[i] or deleted_all[i]:
            continue
        delete = np.logical_and(dists[i] * range_ratio < dists, angles_min[i] - RAD_EPS < angles_min,
                                out=delete)  # Created on first
        np.logical_and(delete, angles_max[i] + RAD_EPS > angles_max, out=delete)
        np.logical_or(deleted_all, delete, out=deleted_all)
    return deleted_all



def get_visible_contour(cnt):
    # cnt of shape (n, 2)
    angles = np.arctan2(cnt[:, 1], cnt[:, 0])
    arg_min, arg_max = np.argmin(angles), np.argmax(angles)
    visible = detection.split.splitContourByIndices(cnt, arg_max, arg_min)[0]
    return visible, angles, (arg_min, arg_max)


def group_by_criteria(array, diff_criteria):
    arg = np.argsort(array)  # Indices of sort
    array = array[arg]
    diff = np.diff(array)  # n-1 array
    res = np.abs(diff) > diff_criteria
    split_indices = np.argwhere(res).reshape((-1,)) + 1
    # print(split_indices)
    # print(np.split(array, split_indices))

    return np.split(arg, split_indices)

    # print(split_indices)
    # print(np.split(arg, split_indices))


def calc_dist(self, cnts):
    angles_cnts = [np.arctan2(cnt[:, 1], cnt[:, 0]) for cnt in cnts]
    min_max_angles_ind = np.array([[np.argmin(angles), np.argmax(angles)] for angles in angles_cnts])