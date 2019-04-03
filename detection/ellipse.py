import cv2


def draw_ellipse(img, ellipse, thickness=1):
    cv2.ellipse(img,
                (int(round(ellipse[0][0])), int(round(ellipse[0][1]))),
                (int(round(ellipse[1][0]/2)), int(round(ellipse[1][1]/2))),
                int(round(ellipse[2])), 0, 360, (255, 0, 255, 255))


def find_ellipse(cnts):
    return [cv2.fitEllipse(cnt) for cnt in cnts]


def scale_ellipse(ellipse_axes, scale_func):
    return