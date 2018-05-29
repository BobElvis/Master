import cv2
import numpy as np
from detection import detection


class Boundary:
    def __init__(self, B, offsets, speed, weight):
        self.n = B.shape[0]
        self.p1 = B
        self.offset = offsets
        self.speed = speed
        self.weight = weight
        self.diff = np.roll(B, -1, axis=0) - B
        self.phi = np.mod(np.arctan2(self.diff[:, 1], self.diff[:, 0]), 2 * np.pi)  # [0, 2*pi>
        self.length = np.sqrt(np.sum(np.square(self.diff), axis=1))
        self.rot_matrix = [np.array(((np.cos(-a), -np.sin(-a)),
                                    (np.sin(-a), np.cos(-a))))
                           for a in np.nditer(self.phi)]

        # Calculate edges:
        angle_edge_next = calc_diff_angle(self.phi)  # [0, 2*pi>
        angle_norm_next = calc_diff_angle(self.phi - offsets) - np.pi  # [-pi, pi>
        self.angle_start_next = self.adjust_start(np.pi / 2 + offsets, angle_norm_next)
        self.angle_start_prev = self.adjust_start(np.pi / 2 - offsets, np.roll(angle_norm_next, 1))
        self.angle_stop_next = np.maximum(angle_edge_next, self.angle_start_next)
        self.angle_stop_prev = np.maximum(np.roll(angle_edge_next, 1), self.angle_start_prev)

    @staticmethod
    def adjust_start(angles, diff):
        concave = diff >= 0
        angles[concave] = angles[concave] + diff[concave] / 2
        return angles

    def is_point_inside(self, x, y):
        c = False
        j = self.n - 1
        for i in range(self.n):
            ix, iy, jx, jy = self.p1[i, 0], self.p1[i, 1], self.p1[j, 0], self.p1[j, 1]
            c1 = (iy > y) != (jy > y)
            c2 = x < (jx - ix) * (y - iy) / (jy - iy) + ix
            if c1 and c2:
                c = not c
            j = i
        return c

    def area(self):
        x, y = self.p1[:, 0], self.p1[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def printInfo(self):
        print("--Corners:")
        print(self.p1)
        print("--Diff:")
        print(self.diff)
        print("--Phi:")
        self.ps(self.phi)
        print("--Length:")
        self.ps(self.length)
        print("--Offsets:")
        self.ps(self.offset)
        print("--Rot matrix")
        for m in self.rot_matrix:
            print(m)
        print("--Start prev:")
        self.ps(self.angle_start_prev)
        print("--Start next:")
        self.ps(self.angle_start_next)
        print("--Stop prev:")
        self.ps(self.angle_stop_prev)
        print("--Stop next:")
        self.ps(self.angle_stop_next)
        print("--Weights:")
        self.ps(self.weight)
        print("--Speed:")
        self.ps(self.speed)

    @staticmethod
    def ps(a):
        print(a.tolist())


def setupBoundary(detect_data, vmax):
    # Settings:
    approx_length = 0.003

    # Create boundary points:
    mask = detect_data.full_mask
    raw_boundary, contours = createBoundary(mask, approx_length)
    b1, b2 = detect_data.scale_measurements(raw_boundary)
    boundary = np.stack((b1, b2), axis=1)
    n = boundary.shape[0]

    # # Define (-1 compared to Matlab):
    # outWest, outEast, outNorth = 26, 45, 9
    # remove = np.array([7, 8, 10, 25])
    #
    # # Offsets:
    offset = np.zeros(n)
    # offset[outNorth] = -5
    # offset[outWest] = 15
    # offset[outEast] = -15
    # offset = np.deg2rad(offset)
    #
    # # Speed:
    speed_max = vmax / 2
    speed = np.ones(n) * speed_max / 2
    # speed[outWest] = speed_max
    # speed[outEast] = speed_max
    # speed[outNorth] = speed_max * 0.8
    # speed[remove] = 0
    #
    #B = Boundary(boundary, offset, speed, None)
    B = Boundary(boundary, offset, speed, None)
    #
    # # Weight:
    weight = np.empty(n)
    weight[:] = np.nan
    # #weight[outWest] = 0.15
    # #weight[outEast] = 0.15
    # #weight[outNorth] = 0.1
    # weight[remove] = 0
    #
    # # Distribute remaining weight:
    remaining_weight = 1 - np.nansum(weight)
    rem_ind = np.isnan(weight)
    weight[rem_ind] = (B.length[rem_ind] / sum(B.length[rem_ind])) * remaining_weight
    B.weight = weight
    return B


def createBoundary(mask, approx_length_factor):
    mask = (~mask).astype('uint8')
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
    cnt = contours[0]
    epsilon = approx_length_factor * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = np.reshape(approx, (len(approx), 2))
    return approx, contours


def drawBoundary(mask, contours, approx_contour):
    cnt = contours[0]
    if len(approx_contour.shape) == 2:
        approx_contour = np.reshape(approx_contour, (len(approx_contour), 1, 2))

    # Drawing img:
    mask_draw = np.zeros((mask.shape[1], mask.shape[0], 3), 'uint8')
    print("Contour approximated from {} to {}".format(len(cnt), len(approx_contour)))
    cv2.drawContours(mask_draw, contours, -1, (0, 255, 0), 1)
    cv2.drawContours(mask_draw, [approx_contour], 0, (0, 0, 255), 1)
    cv2.imshow('image', mask_draw[400:1000, 100:900, :])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calc_diff_angle(angles):
    diff_angles = np.roll(angles, -1) - angles
    return np.mod(2 * np.pi - np.mod(diff_angles + np.pi, 2 * np.pi), 2 * np.pi)  # [0, 2*pi>


def polyskel(boundary, img):
    from polyskel.polyskel import skeletonize
    boundary = np.reshape(boundary, (boundary.shape[0], boundary.shape[2]))
    boundary_list = boundary.tolist()
    skeleton = skeletonize(boundary_list)
    print(boundary_list)
    for res in skeleton:
        print(res)
    for idx, arc in enumerate(skeleton):
        # if idx < len(boundary):
        #    continue
        for sink in arc.sinks:
            if [sink.x, sink.y] in boundary_list:
                continue
            cv2.line(img, (int(arc.source.x), int(arc.source.y)), (int(sink.x), int(sink.y)), (0, 255, 0), 2)
