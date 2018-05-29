import cv2
import util
from model.model import *
from detection.split import *
import detection.geometry as geometry
import dataloader
import os
from detection.dataset import Dataset, find_files
from detection.dataconfigs import DataConfig


AVG_FOLDER = 'land'
RAW_SIZE = 1024
SAT_PATH = "land/landSat190ext.png"
SAT_RANGE = 190


def save_avg_mask(mask, dataset: Dataset, p, i_start, i_end):
    d = dataset
    filename = "AVG(rot{},r{})({},{})({},{})". \
        format(d.config.rotation,d.config.radar_range, p, d.config.day, d.config.partition, i_start, i_end)
    path = os.path.join(AVG_FOLDER, filename)
    util.saveImg(mask, path, 'png')


def get_avg_mask(data_config: DataConfig):
    files = find_files(AVG_FOLDER, 'png')
    criteria = "AVG(rot{},r{}".format(data_config.rotation, data_config.radar_range)
    masks = [f for f in files if f.startswith(criteria)]
    if len(masks) == 1:
        return util.readMask(os.path.join(AVG_FOLDER, masks[0] + ".png"))
    else:
        print("Error locating mask. Number of masks found for config: {}".format(data_config))


def get_land_mask(data_config: DataConfig):
    return util.get_mask(data_config.radar_range, RAW_SIZE, SAT_PATH, SAT_RANGE)


def contourCentroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


class DetectData:
    def __init__(self, data_config):
        self.measScale = 2 * data_config.radar_range / RAW_SIZE
        self.sat_mask = get_land_mask(data_config)
        self.avg_mask = get_avg_mask(data_config)
        self.full_mask = np.logical_or(self.sat_mask, self.avg_mask)

    def scale_measurements(self, measurements):
        # ------ Scaling measurements.
        mx = (measurements[:, 0] - RAW_SIZE / 2) * self.measScale
        my = -(measurements[:, 1] - RAW_SIZE / 2) * self.measScale
        return mx, my


class DetectionBase(dataloader.DataSource):
    def __init__(self, dataset, rotation):
        self.dataset = dataset
        rotMatrix = cv2.getRotationMatrix2D((RAW_SIZE / 2 - 0.5, RAW_SIZE / 2 - 0.5), rotation, 1)
        self.rotMatrixInverse = cv2.invertAffineTransform(rotMatrix)

    def load_radar(self, idx):
        radar_img = cv2.imread(self.dataset[idx] + ".png", cv2.IMREAD_COLOR)  # Read in BGR
        radar_img = radar_img[:, :, 1]  # Extract green channel. Values 0 or 128.
        return self.rotateRadar(radar_img)

    def load_camera(self, idx):
        return util.readImg(self.dataset[idx] + ".jpg")

    def load_raw_data(self, idx):
        return self.load_radar(idx), self.load_camera(idx)

    def load_data(self, idx):
        radar_img, camera_img = self.load_raw_data(idx)
        scan = self.detect_radar(radar_img)
        scan.time = idx
        scan.camera_img = self.detect_camera(camera_img)
        return scan

    def detect_radar(self, radar_img) -> Scan:
        raise NotImplementedError("Not implemented")

    def detect_camera(self, camera_img):
        return camera_img

    def rotateRadar(self, radar_img):
        return cv2.warpAffine(radar_img, self.rotMatrixInverse, dsize=radar_img.shape,
                              flags=(cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP))

    def __len__(self):
        return len(self.dataset)


class Detection(DetectionBase):
    def __init__(self, dataset, detect_data, meas_init, resize=None):
        super().__init__(dataset, detect_data.rotation)

        self.dd = detect_data
        self.resize = 1 if resize is None else resize
        self.meas_init = meas_init
        self.debug = False

        # Parameters for splitting:
        self.enable_splitting = False
        self.areaMin = 150
        self.areaMinCnt = self.areaMin * 0.9
        self.ratio_min = 0.2 #0.1  # 0.2
        self.ratio_sum_min = 0.4  # 0.55
        self.ellipse_ratio_min = 3.5

    def detect_radar(self, radar_img):
        # camera_img: 3-channel img.
        # radar_img: 1-channel img.
        # Outputs scan. Measurements are sorted in y-coordinate (from largest to smallest)

        #radar_img[self.dd.full_mask] = 0

        # ----- Draw raw output:
        drawImg = np.zeros((radar_img.shape[0], radar_img.shape[1], 4), dtype='uint8')
        _, cnts, _ = cv2.findContours(radar_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(drawImg, cnts, -1, (255, 255, 255, 100), cv2.FILLED)

        # ----- Find contours, filter on size:
        cnts = [c for c in cnts if cv2.contourArea(c) > self.areaMinCnt]
        measurements = self.detectCnt(cnts, drawImg)
        measurements = np.array(measurements)
        if len(measurements) == 0:
            measurements = np.empty((0, 2))

        # ------ Scaling measurements:
        mx, my = self.dd.scale_measurements(measurements)
        return Scan(mx, my, drawImg)

    def detect_camera(self, camera_img):
        camera_img = camera_img[400:-150, :, :]  # Crop top/bottom off the camera.
        camera_img = cv2.resize(camera_img, dsize=(0, 0), fx=self.resize, fy=self.resize)  # Resize
        camera_img = self.improve_camera(camera_img)
        return camera_img

    @staticmethod
    def get_best_defect_simple(cnt):
        # Get defects and sort by defect size (largest to smallest):
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
        if defects is None:
            return False, None, None
        defects = np.reshape(defects, (defects.shape[0], 4))
        defects = defects[defects[:, 3].argsort()[::-1]]
        if defects.shape[0] < 2:
            return False, None, None
        return True, defects[0], defects[1]

    @staticmethod
    def get_best_defect(cnt):
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
            if res[0] is False or not self.enable_splitting:
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

    def improve_camera(self, img):
        # Gamma correction:
        gamma = 2
        img = np.power(img/255, 1/gamma)*255
        img = img.astype('uint8')

        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        lum, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        cl = clahe.apply(lum)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
