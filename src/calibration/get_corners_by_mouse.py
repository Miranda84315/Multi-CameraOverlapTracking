import cv2
import numpy as np

image_root = 'D:/Code/MultiCamOverlap/dataset/calibration/Player05/cam'
matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/Player05/information/temp_corner'
global refPt
get_Corner = False
get_ROI = True


def get_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        refPt.append([x, y])


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
