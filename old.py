# # Testing ellipse:
# im2, contours, hierarchy = cv2.findContours(radar_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# for idx in range(0, len(contours)):
#     cnt = contours[idx]
#     if len(cnt) < 5 or cv2.contourArea(cnt) < self.areaMin:
#         continue
#     cnt = contours[idx]
#     ellipse = cv2.fitEllipse(cnt)  # (x,y), (Ma, ma), angle
#     earea = ellipse[1][0] * ellipse[1][1] * math.pi / 4
#     carea = cv2.contourArea(cnt)
#     print(earea)
#     print(carea)
#     print(earea / carea)
