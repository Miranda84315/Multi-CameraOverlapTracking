import cv2
import numpy as np
from matplotlib.path import Path

image_root = 'D:/Code/MultiCamOverlap/dataset/calibration/Player05/cam'
matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/Player05/information/temp_corner'
global refPt
get_Corner = False
get_ROI = False
test_ROI = True

p1 = Path([(321, 685), (1605, 644), (1918, 731), (1914, 1075), (0, 1080), (0, 794)])
p2 = Path([(2, 558), (795, 520), (1858, 677), (1691, 1077), (0, 1075)])
p3 = Path([(0, 455), (792, 388), (1905, 738), (1391, 1077), (0, 1072)])
p4 = Path([(51, 478), (462, 1074), (811, 1075), (1732, 658), (921, 484)])


def get_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        refPt.append([x, y])
    # cv2.circle(img, (x, y), 7, (255, 255, 0), -1)


if get_Corner:
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', get_coordinate)
        refPt = []
        print('------------------cam: ' + str(icam))
        while(len(refPt) < 16):
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        cv2.destroyAllWindows()

if get_ROI:
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1344, 756)
        cv2.setMouseCallback('image', get_coordinate)
        refPt = []
        print('------------------cam: ' + str(icam))
        while(1):
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        cv2.destroyAllWindows()


def inROI(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        p = [p1, p2, p3, p4]
        print(p[icam-1].contains_points([(x, y)])[0])


if test_ROI:
    for icam in range(1, 5):
        image_file = image_root + str(icam) + '.jpg'
        img = cv2.imread(image_file)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1344, 756)
        cv2.setMouseCallback('image', inROI)
        refPt = []
        print('------------------cam: ' + str(icam))
        while(1):
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 按q键退出
                break
        cv2.destroyAllWindows()
        