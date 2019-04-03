import util
from detection.geometry import contour_centroid
from model.model import *
from detection.split import *
import dataloader
import os
from dataset import DatasetFolder, find_files, filename_to_date
from detection.dataconfigs import DataConfig
from detection.ghosts import mark_multiple

AVG_FOLDER = 'land'
RAW_SIZE = 1024
SAT_PATH = "land/landSat190ext.png"
SAT_RANGE = 190


def save_avg_mask(mask, dataset: DatasetFolder, p, i_start, i_end):
    d = dataset
    filename = "AVG(rot{},r{})({})(p{})({},{})". \
        format(d.config.rotation, d.config.radar_range, d.folder_name, p, i_start, i_end)
    path = os.path.join(AVG_FOLDER, filename)
    util.saveImg(mask, path, 'png')


def get_avg_mask(data_config: DataConfig):
    files = find_files(AVG_FOLDER, 'png')
    criteria = "AVG(rot{},r{}".format(data_config.rotation, data_config.radar_range)
    masks = [f for f in files if f.startswith(criteria)]
    if len(masks) == 0:
        raise FileNotFoundError("No masks found for config: {}".format(data_config))
    elif len(masks) == 1:
        return util.readMask(os.path.join(AVG_FOLDER, masks[0] + ".png"))
    else:
        print("WARNING. Multiple average masks found for config: {}".format(data_config))
        return masks[0]


def create_land_mask(data_config: DataConfig):
    landsat_mask = get_land_mask(data_config)
    range_mask = util.create_range_mask(190, data_config.radar_range, landsat_mask.shape[0])
    ext = max(190, data_config.radar_range)
    landsat_mask[range_mask] = True
    landsat_mask, new_extent = util.shrink_mask(~landsat_mask, extent=(-ext, ext, -ext, ext))
    return landsat_mask, new_extent


def get_land_mask(data_config: DataConfig):
    land_mask = util.get_mask(data_config.radar_range, RAW_SIZE, SAT_PATH, SAT_RANGE)
    range_mask = util.create_range_mask(data_config.radar_range * 0.95, SAT_RANGE, RAW_SIZE)
    land_mask[range_mask] = True
    return land_mask


def get_shore_mask(full_mask, range_meter, meter_pixel_ratio):
    pixel_range = int(round(range_meter / meter_pixel_ratio))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_range, pixel_range))
    dilated = cv2.dilate(full_mask.astype(np.uint8), element) > 0
    #print(dilated.shape)
    dilated[full_mask] = False
    return dilated


class DetectData:
    def __init__(self, data_config: DataConfig):
        self.meter_pixel_ratio = 2 * data_config.radar_range / RAW_SIZE
        self.sat_mask = get_land_mask(data_config)
        self.avg_mask = get_avg_mask(data_config)
        self.full_mask = np.logical_or(self.sat_mask, self.avg_mask)
        self.data_config = data_config
        self.shore_filter_range = 0
        self.shore_mask = None if self.shore_filter_range <= 0 else \
            get_shore_mask(self.full_mask, self.shore_filter_range, self.meter_pixel_ratio)

    def scale_measurements(self, measurements, reverse=False):
        # ------ Scaling measurements.
        if reverse:
            mx = measurements[:, 0] / self.meter_pixel_ratio + RAW_SIZE / 2
            my = -measurements[:, 1] / self.meter_pixel_ratio + RAW_SIZE / 2
        else:
            mx = (measurements[:, 0] - RAW_SIZE / 2) * self.meter_pixel_ratio
            my = -(measurements[:, 1] - RAW_SIZE / 2) * self.meter_pixel_ratio
        return np.stack((mx, my), axis=1)

    def scale_sizes(self, sizes):
        return sizes * self.meter_pixel_ratio

    def area(self):
        sum_pixel2 = np.sum(np.logical_not(self.full_mask))
        return sum_pixel2*(self.meter_pixel_ratio ** 2)


class DetectionBase(dataloader.DataSource):
    def __init__(self, dataset: DatasetFolder):
        self.dataset = dataset
        c = dataset.config
        rotMatrix = cv2.getRotationMatrix2D((c.radar_img_size / 2 - 0.5, c.radar_img_size / 2 - 0.5), c.rotation, 1)
        self.rotMatrixInverse = cv2.invertAffineTransform(rotMatrix)

    def load_radar(self, idx):
        radar_img = cv2.imread(self.dataset[idx] + ".png", cv2.IMREAD_COLOR)  # Read in BGR
        radar_img = radar_img[:, :, 1]  # Extract green channel. Values 0 or 128.
        return self.rotateRadar(radar_img)

    def get_camera_path(self, idx):
        return self.dataset[idx] + ".jpg"

    def load_camera(self, idx):
        return util.readImg(self.get_camera_path(idx))

    def load_raw_data(self, idx, camera=True):
        return self.load_radar(idx), self.load_camera(idx)

    def load_data(self, idx):
        radar_img = self.load_radar(idx)
        scan = self.detect_radar(radar_img)
        scan.idx = idx
        scan.time = filename_to_date(self.dataset.files[idx])
        scan.camera_path = self.get_camera_path(idx)
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


class Detector(DetectionBase):
    def __init__(self, dataset, detect_data: DetectData, area_min, area_multiple, range_multiple):
        super().__init__(dataset)

        self.dd = detect_data
        self.resize = 1

        # Parameters
        self.areaMinCnt = area_min
        self.area_multiple = area_multiple
        self.range_multiple = range_multiple
        self.filter_multiples = area_multiple > 0

        # Used for estimating error of measurments.
        self.estimate_error = False
        self.error = [] if self.estimate_error else None
        self.multiple = []  # Array indicating if a multiple was found.

    def detect_radar(self, radar_img):
        # camera_img: 3-channel img.
        # radar_img: 1-channel img.
        # Outputs scan. Measurements are sorted in y-coordinate (from largest to smallest)

        radar_img[self.dd.full_mask] = 0

        # Finding contours. Orientation is always counter-clockwise.
        _, cnts, _ = cv2.findContours(radar_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # ----- Draw raw output:
        drawImg = np.zeros((radar_img.shape[0], radar_img.shape[1], 4), dtype='uint8')
        cv2.drawContours(drawImg, cnts, -1, (255, 255, 255, 100), cv2.FILLED)

        # ----- Find contours, filter on size, then sort on size:
        cnts_area = [(c, cv2.contourArea(c)) for c in cnts]
        cnts_area = [e for e in cnts_area if e[1] > self.areaMinCnt]
        cnts_area = sorted(cnts_area, key=lambda x: x[1])

        # ----- Extracting measurements:
        data = [(contour_centroid(cnt), cnt, area) for cnt, area in cnts_area]

        # ---- Shore filter:
        if self.dd.shore_mask is not None:
            data = [d for d in data if not self.dd.shore_mask[d[0][1], d[0][0]]]
            for d in data:
                print("M: {}, {}: {}".format(d[0][0], d[0][1], self.dd.shore_mask[d[0][0], d[0][1]]))

        # --- Extract:
        measurements = [m for m, cnt, area in data]
        cnts = [cnt for m, cnt, area in data]
        area = [area for m, cnt, area in data]

        # --- Scale:
        measurements = np.array(measurements)
        if len(measurements) == 0:
            measurements = np.empty((0, 2))
        measurements = self.dd.scale_measurements(measurements)

        # ------ Finding ellipses:
        ellipses = None
        # ellipses = [cv2.fitEllipse(cnt) for cnt in cnts]
        # for e in ellipses:
        #    draw_ellipse(drawImg, e)
        # axes = self.dd.scale_sizes(np.array([e[1] for e in ellipses])) / 2
        # angle = np.array([e[2] for e in ellipses])
        # ellipses = np.column_stack((axes.reshape(-1, 2), angle.reshape(-1, 1)))

        # ------ Finding multiples ------------:
        if self.filter_multiples:
            removed = mark_multiple(self.dd, measurements, cnts, area, self.area_multiple, self.range_multiple)
            self.multiple.append(removed is not None)
        else:
            removed = None

        # ----- Create detection objects:
        detections = []
        for i in range(measurements.shape[0]):
            if removed is None or not removed[i]:
                detections.append(Detection(measurements[i, :], area[i], cnts[i]))

        # ----- Removing shadowed measurements:
        if removed is not None:
            measurements = measurements[np.logical_not(removed)]

        return Scan(drawImg, detections, measurements)

    def detect_camera(self, camera_img):
        camera_img = camera_img[400:-150, :, :]  # Crop top/bottom off the camera.
        camera_img = cv2.resize(camera_img, dsize=(0, 0), fx=self.resize, fy=self.resize)  # Resize
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
                measurements.append(contour_centroid(cnt))
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
            elif ellipse_ratio > self.ellipse_ratio_min and areaCnt / 2 > self.areaMinCnt:  # and level == 0:
                e_m1, e_m2 = splitEllipse(ellipse)
                measurements.append(e_m1)
                measurements.append(e_m2)
            else:
                measurements.append(contour_centroid(cnt))
        return measurements
