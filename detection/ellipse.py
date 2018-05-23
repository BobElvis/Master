import cv2

def draw_ellipse(img, ellipse, thickness=1):
    cv2.ellipse(img,
                (int(round(ellipse[0][0])), int(round(ellipse[0][1]))),
                (int(round(ellipse[1][0]/2)), int(round(ellipse[1][1]/2))),
                int(round(ellipse[2])), 0, 180, (255, 0, 255, 255))


def ellipseError(contour, ellipse):
    # Contour must been found with APPROX_NONE. Ellipse is the rotated bounding box.
    print("-- Ellipse fit --")
    area = cv2.contourArea(contour)
    length = len(contour)
    print("Contour. Length: {}. Area: {}".format(len(contour), cv2.contourArea(contour)))

    cx, cy = ellipse[0]
    a, b = ellipse[1]
    angle = ellipse[2]/180*math.pi
    print(ellipse)

    error = 0
    errNeg = 0
    errPos = 0
    for i in range(0, len(contour)):
        conx, cony = contour[i][0], contour[i][1]
        posx = (conx - cx) * cos(-angle) - (cony - cy) * sin(-angle)
        posy = (conx - cx) * cos(-angle) + (cony - cy) * sin(-angle)
        err = (posx/a)**2 + (posy/b)**2 - 1
        if err < 0:
            errNeg -= err
        else:
            errPos += err
    error = errPos + errNeg
    print("E: {:.4g}. N: {:.4g} + P: {:.4g}. E/A: {:.4g}. E/L: {:.4g}".format(error, errNeg, errPos, error/area, error/length))