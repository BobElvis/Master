import cv2
import numpy as np
import math


def splitContourByIndices(cnt, i1, i2):
    # Expect cnt to be of shape=(length, 1, 2) (does not matter)
    # c1 = i1 -> i2, c2 = i2 -> i1
    flipped = i1 > i2
    if flipped:
        i1, i2 = i2, i1
    c1 = cnt[i1:(i2 + 1)]
    c2 = np.concatenate((cnt[:(i1+1)], cnt[i2:]), axis=0)
    if flipped:
        return c2, c1
    else:
        return c1, c2


def splitEllipse(ellipse):
    c = ellipse[0]
    dc = ellipse[1][1]/6
    angle = ellipse[2]*math.pi/180 + math.pi/2
    m1 = c[0] + dc*math.cos(angle), c[1] + dc*math.sin(angle)
    m2 = c[0] - dc*math.cos(angle), c[1] - dc*math.sin(angle)
    return m1, m2


class ConvexSplitter:
    def split(self, cnt):
        cntHull = [cv2.convexHull(c, returnPoints=True) for c in cnt]

        areaCnt = [cv2.contourArea(c) for c in cnt]
        areaHull = [cv2.contourArea(c) for c in cntHull]

        for i in range(0, len(cnt)):
            if areaCnt[i] == 0:
                continue
            print("Area: cnt={:.4g}, convex={:.4g}, ratio={:.4g}".format(areaCnt[i], areaHull[i], areaHull[i]/areaCnt[i]))

class SplitErosion:
    def __init__(self, kernel, size):
        self.k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#kernel
        self.k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_flag = True
        self.size = size
        self.labels_storage = np.zeros(self.size, 'uint8')

    def connectedComponents(self, img):
        return cv2.connectedComponents(img, labels=self.labels_storage, connectivity=8)

    def numCC(self, img):
        #outCC = self.connectedComponents(img)
        #return outCC[0] - 1
        return len(self.findContour(img))

    def erode(self, bin_img):
        if self.kernel_flag:
            cv2.erode(bin_img, self.k_ellipse, dst=bin_img)
        else:
            cv2.erode(bin_img, self.k_rect, dst=bin_img)
        #self.kernel_flag = ~self.kernel_flag

    @staticmethod
    def findContour(img):
        _, cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cnt

    def split(self, bin_img):
        outCC = self.connectedComponents(bin_img.astype('uint8'))
        imgs = [(outCC[1] == i).astype('uint8') for i in range(1, outCC[0])]
        doneImgs = []
        while len(imgs) > 0:
            img = imgs.pop()
            outCC = self.split_aux(img)
            new_imgs = [(outCC[1] == i).astype('uint8') for i in range(1, outCC[0])]
            if len(new_imgs) == 0:
                doneImgs.append(img)
            imgs.extend(new_imgs)
        return doneImgs

    def split_aux(self, single_CC_img):
        eroded_img = single_CC_img.copy()
        nCC = 1
        outCC = None
        i = 0
        while nCC == 1:
            self.erode(eroded_img)
            outCC = self.connectedComponents(eroded_img)
            nCC = outCC[0] - 1
            i += 1
        print("Stopped at i={}, num={}.".format(i, nCC))

        #if nCC > 1:
        #    self.erode(eroded_img)
        #    print(" -> After another erode: {}".format(self.numCC(eroded_img)))

        return outCC