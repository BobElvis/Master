import cv2
import util
from model.model import *
from detection.split import *
import detection.geometry as geometry
import dataloader


def get_mask(range, resolution, path, mask_range):
    mask = util.readImgGray(path)
    s = range / mask_range
    c = round(mask.shape[0] / 2)
    r = round(c * s)
    maskCrop = mask[c - r:c + r, c - r:c + r]
    return cv2.resize(maskCrop, (resolution, resolution), interpolation=cv2.INTER_NEAREST) > 0


def getAvgMask():
    return util.readMask("land/landAvg.png")


def contourCentroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


class DetectData:
    def __init__(self):
        self.rawSize = 1024
        self.range = 175
        self.measScale = 2 * self.range / self.rawSize
        self.rotation = 183
        self.sat_mask = get_mask(self.range, self.rawSize, "land/landSat190ext2.png", 190)
        self.avg_mask = util.readMask("land/landAvg.png")
        self.full_mask = np.logical_or(self.sat_mask, self.avg_mask)

    def scale_measurements(self, measurements):
        # ------ Scaling measurements.
        mx = (measurements[:, 0] - self.rawSize / 2) * self.measScale
        my = -(measurements[:, 1] - self.rawSize / 2) * self.measScale
        return mx, my


class Detection(dataloader.DataSource):
    def __init__(self, dataset, detect_data, meas_init, resize=None):
        self.dataset = dataset

        self.dd = detect_data
        rotMatrix = cv2.getRotationMatrix2D((self.dd.rawSize / 2 - 0.5, self.dd.rawSize / 2 - 0.5), self.dd.rotation, 1)
        self.rotMatrixInverse = cv2.invertAffineTransform(rotMatrix)
        self.resize = 1 if resize is None else resize
        self.meas_init = meas_init
        self.debug = False

        # Parameters for splitting:
        self.areaMin = 150
        self.areaMinCnt = self.areaMin * 0.9
        self.ratio_min = 0.1  # 0.2
        self.ratio_sum_min = 0.4  # 0.55
        self.ellipse_ratio_min = 3.5

    def load_data(self, idx):
        path = self.dataset[idx]
        camera_img = util.readImg(path + ".jpg")
        radar_img = cv2.imread(path + ".png", cv2.IMREAD_COLOR)  # Read in BGR
        radar_img = radar_img[:, :, 1]  # Extract green channel. Values 0 or 128.
        return self.detect(camera_img, radar_img)

    def __len__(self):
        return len(self.dataset)

    def detect(self, camera_img, radar_img):
        # camera_img: 3-channel img.
        # radar_img: 1-channel img.
        # Outputs scan. Measurements are sorted in y-coordinate (from largest to smallest)

        radar_img = self.rotateRadar(radar_img)
        radar_img[self.dd.full_mask] = 0

        # ----- Draw raw output:
        drawImg = np.zeros((radar_img.shape[0], radar_img.shape[1], 4), dtype='uint8')
        _, cnts, _ = cv2.findContours(radar_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(drawImg, cnts, -1, (255, 255, 255, 255), cv2.FILLED)

        # ----- Find contours, filter on size:
        cnts = [c for c in cnts if cv2.contourArea(c) > self.areaMinCnt]
        measurements = self.detectCnt(cnts, drawImg)
        measurements = np.array(measurements)
        if len(measurements) == 0:
            measurements = np.empty((0, 2))

        # ------ Scaling measurements:
        mx, my = self.dd.scale_measurements(measurements)

        # ------ Camera:
        camera_img = camera_img[400:, :, :]  # Crop top off the camera.
        camera_img = cv2.resize(camera_img, dsize=(0, 0), fx=self.resize, fy=self.resize)  # Resize
        return Scan(0, drawImg, camera_img, mx, my)

    def rotateRadar(self, radar_img):
        return cv2.warpAffine(radar_img, self.rotMatrixInverse, dsize=radar_img.shape,
                              flags=(cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP))

    def get_best_defect_simple(self, cnt):
        # Get defects and sort by defect size (largest to smallest):
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
        defects = np.reshape(defects, (defects.shape[0], 4))
        defects = defects[defects[:, 3].argsort()[::-1]]
        if defects.shape[0] < 2:
            return False
        return True, defects[0], defects[1]

    def get_best_defect(self, cnt):
        # Not done...
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
        defects = np.reshape(defects, (defects.shape[0], 4))
        cnt = np.reshape(cnt, (cnt.shape[0], 2))
        n_def = defects.shape[0]
        if n_def < 2:
            return False
        s1 = cnt[defects[:, 0]]
        s2 = cnt[defects[:, 1]]
        cnt_def = [splitContourByIndices(cnt, defects[i, 0], defects[i, 1])[0] for i in range(n_def)]
        cnt_def = [np.reshape(c, (c.shape[0], 2)) for c in cnt_def]
        def_lens = [geometry.distance_to_line_np(s1[i, :], s2[i, :], cnt_def[i]) for i in range(n_def)]


    def detectCnt(self, cnts, draw_img, level=0):
        measurements = []
        indentStr = "  " * level
        for cnt in cnts:

            cntConvex = cv2.convexHull(cnt, returnPoints=True)
            areaCnt = cv2.contourArea(cnt)

            if self.debug:
                print("{}{}Contour (A:{:.0f}):".format(indentStr, level, areaCnt))
                cv2.drawContours(draw_img, [cntConvex], 0, (255, 0, 0, 255), thickness=1)

            res = self.get_best_defect_simple(cnt)
            if res[0] is False:
                measurements.append(contourCentroid(cnt))
                continue

            def1, def2 = res[1], res[2]
            def1dist, def2dist = def1[3] / 256., def2[3] / 256.
            c1, c2 = splitContourByIndices(cnt, def1[2], def2[2])
            area_c1 = cv2.contourArea(c1)
            area_c2 = cv2.contourArea(c2)
            dist_defects = geometry.distance(cnt[def1[2]][0], cnt[def2[2]][0])

            # Ratios:
            dist_ratio1 = def1dist / dist_defects
            dist_ratio2 = def2dist / dist_defects
            dist_ratio_sum = dist_ratio1 + dist_ratio2

            split_area = area_c1 > self.areaMinCnt and area_c2 > self.areaMinCnt
            split_ratio = dist_ratio1 > self.ratio_min and \
                          dist_ratio2 > self.ratio_min and \
                          dist_ratio_sum > self.ratio_sum_min

            # Ellipse ratio:
            ellipse = cv2.fitEllipse(cnt)
            ellipse_ratio = ellipse[1][1] / ellipse[1][0]  # a/b

            # Debugging:
            if self.debug:
                print("{} - A1: {:.1f}, A2: {:.1f} /{:.1f}, "
                      "r: {:.1f},{:.1f}/{:.1f}, r_sum: {:.1f}".
                      format(indentStr, area_c1, area_c2, self.areaMinCnt,
                             dist_ratio1 * 100, dist_ratio2 * 100, self.ratio_min * 100,
                             dist_ratio_sum * 100, self.ratio_sum_min * 100))
                print("{} - Ellipse: a/b = {:.2f}/{:.2f}".
                      format(indentStr, ellipse_ratio, self.ellipse_ratio_min))

            # Determine what to do further:
            if split_ratio and split_area:
                measurements.extend(self.detectCnt([c1, c2], draw_img, level + 1))
                cv2.drawContours(draw_img, [c1, c2], -1, (0, 255, 0, 255), thickness=1)
            elif ellipse_ratio > self.ellipse_ratio_min and areaCnt/2 > self.areaMinCnt: #and level == 0:
                e_m1, e_m2 = splitEllipse(ellipse)
                measurements.append(e_m1)
                measurements.append(e_m2)
            else:
                measurements.append(contourCentroid(cnt))
        return measurements
