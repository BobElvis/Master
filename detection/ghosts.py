import numpy as np
import detection.split
import cv2
from detection.geometry import contour_centroid


def mark_multiple(dd, m, cnts, draw_img=None):
    n = len(cnts)
    if n < 2:
        return
    dist_centroid = np.linalg.norm(m, axis=1)
    dist_argsort = np.argsort(dist_centroid)
    d = dist_centroid[dist_argsort]

    multiple = np.round(d[1:]/d[0])
    closest_match = multiple*d[0]
    print(closest_match)
    print(d[1:])
    rel_diff = closest_match/d[1:] - 1
    print(rel_diff)
    print(d[1:]/closest_match - 1)

    shadowed = np.logical_and(np.abs(rel_diff) < 0.1, multiple > 1)
    return dist_argsort[1:][shadowed]



    return

    # cnts should be (n, 2).
    if len(cnts) < 1:
        return

    visible_cnts = [get_visible_contour(cnt)[0] for cnt in cnts]
    draw_cnts = [dd.scale_measurements(cnt, reverse=True).reshape((-1, 1, 2)).astype(np.int32)
                 for cnt in visible_cnts]
    visible_center = np.array([contour_centroid(cnt) for cnt in draw_cnts])
    print(visible_center)

    if draw_img is not None and True:
        #draw_cnts = [dd.scale_measurements(cnt, reverse=True).reshape((-1, 1, 2)).astype(np.int32)
        #             for cnt in visible_cnts]
        cv2.drawContours(draw_img, draw_cnts, -1, (0, 255, 0, 255), thickness=1)
        #centers = [contour_centroid(draw_cnt) for draw_cnt in draw_cnts]
        for center in visible_center:
            #print(np.linalg.norm(dd.scale_measurements(np.array(center).reshape((-1, 2)))))
            cv2.circle(draw_img, (center[0], center[1]), 3, (255, 0, 0, 255))

    visible_center = dd.scale_measurements(visible_center)
    dist = np.linalg.norm(visible_center, axis=1)
    dist_argsort = np.argsort(dist)
    dist = dist[dist_argsort]
    ratios = dist[1:]/dist[0]
    print(ratios)
    print(ratios - np.round(ratios))

    angle_thresh = 5

    if m.shape[0] < 2:
        return

    angles_center = np.arctan2(m[:, 1], m[:, 0]) * 180 / np.pi
    groups_arg = group_by_criteria(angles_center, angle_thresh)
    for group_arg in groups_arg:
        if len(group_arg) < 2:
            continue
        # Determine if multiple:
        m_group = m[group_arg]
        dist = np.linalg.norm(m_group, axis=1)
        dist_argsort = np.argsort(dist)
        dist_sort = dist[dist_argsort]
        ratios = dist_sort[1:] / dist_sort[0]
        print(ratios)
        print(ratios - np.round(ratios))

    return

    # Scale the contours:
    cnts = [self.dd.scale_measurements(cnt.reshape((-1, 2))) for cnt in cnts_raw]

    # Calculate min-max angles:
    angles_cnts = [np.arctan2(cnt[:, 1], cnt[:, 0]) for cnt in cnts]
    min_max_angles = np.array([[np.amin(angles), np.amax(angles)] for angles in angles_cnts])
    min_max_angles *= (180 / np.pi)
    min_max_angles = np.abs(min_max_angles)

    if m.shape[0] > 1:
        angles = np.arctan2(m[:, 1], m[:, 0])
        angles_deg = angles * 180 / np.pi

        dist1 = np.linalg.norm(m, axis=1)
        arg = self.calc_ratio(dist1)
        ang_sort = min_max_angles[arg, :]
        diff = ang_sort[1:] - ang_sort[0, :]
        shadow = np.logical_and(diff[:, 0] < 0, diff[:, 1] > 0)
        shadowed_indices = arg[1:][shadow]
        return shadowed_indices
    else:
        return None


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