import numpy as np
import cv2


def mask2img(mask, color):
    r, g, b, a = color
    shape = (mask.shape[0], mask.shape[1])
    rm = np.ones(shape, dtype=np.float) * r
    gm = np.ones(shape, dtype=np.float) * g
    bm = np.ones(shape, dtype=np.float) * b
    am = np.array(mask, dtype=np.float) * a
    return np.stack((rm, gm, bm, am), axis=2)


def blend(background, foreground, alpha):
    alpha = np.stack((alpha, alpha, alpha), axis=2)
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    return cv2.add(foreground, background)


class Background:
    def __init__(self, radar_range, x_lim, y_lim, out_size):
        self.imgs = []
        self.out_size = out_size
        self.radar_range = radar_range
        self.x_lim = x_lim
        self.y_lim = y_lim

    def add_image(self, img, extent, alpha=1.):
        img = img.astype(float) / np.amax(img)
        img = np.dstack((img, np.full(img.shape[0:2], alpha)))
        self.__add_image__(img, extent)

    def add_overlay(self, mask, color, extent):
        self.__add_image__(mask2img(mask, color), extent)

    def __add_image__(self, img, extent):
        if not isinstance(extent, (tuple, list)):
            extent = (-extent, extent, -extent, extent)
        self.imgs.append((img, extent))

    def get_img(self):
        # Dimensions in meters:
        out_x1 = min(self.imgs, key=lambda x: x[1][0])[1][0]
        out_x2 = max(self.imgs, key=lambda x: x[1][1])[1][1]
        out_y1 = min(self.imgs, key=lambda x: x[1][2])[1][2]
        out_y2 = max(self.imgs, key=lambda x: x[1][3])[1][3]
        out_extent = [out_x1, out_x2, out_y1, out_y2]
        out_w = out_x2 - out_x1
        out_h = out_y2 - out_y1

        # Dimensions in pixels:
        out_wp = int(round(out_w/max(out_w, out_h) * self.out_size))
        out_hp = int(round(out_h/max(out_w, out_h) * self.out_size))

        # Pixel/meter ratio:
        dx = out_wp/out_w
        dy = out_hp/out_h

        new_img_list = []

        # Create padding and resize
        for img, extent in self.imgs:
            w, h = extent[1] - extent[0], extent[3] - extent[2]
            size_p = (int(round(out_wp*w/out_w)), int(round(out_hp*h/out_h)))
            img = cv2.resize(img, (size_p[1], size_p[0]), interpolation=cv2.INTER_CUBIC)

            px1 = int(round((extent[0]-out_x1)*dx))
            py1 = int(round((extent[2]-out_y1)*dy))
            px2 = out_wp - px1 - size_p[0]
            py2 = out_hp - py1 - size_p[1]

            img = np.pad(img, ((px1, px2), (py1, py2), (0, 0)), 'constant')
            assert img.shape[0] == out_wp and img.shape[1] == out_hp, \
                "{}:({}:{})".format(img.shape, out_wp, out_hp)
            new_img_list.append(img)

        # Blend:
        bg = new_img_list[0][:, :, :3]
        for i in range(1, len(new_img_list)):
            overlay = new_img_list[i]
            bg = blend(bg, overlay[:, :, :3], overlay[:, :, 3])
        bg = np.dstack((bg, new_img_list[0][:, :, 3]))
        bg -= np.amin(bg)
        bg /= np.amax(bg)
        return bg, out_extent
        #return (bg * 255).astype('uint8'), out_extent




