import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
'''
use k-means to combined detections
'''

experiment_root = 'D:/Code/MultiCamOverlap/experiments/demo/'
detection_root = 'D:/Code/MultiCamOverlap/dataset/detections/cam'
matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/information/'

start_time = [1, 1, 1, 1]
NumFrames = [225, 225, 225, 225]
PartFrames = [[225, 225, 225, 225]]
cam_num = 4
width = 1920
height = 1080


def load_detection(cam_num):
    detections = []
    for icam in range(1, cam_num + 1):
        load_file = detection_root + str(icam) + '.mat'
        data = scipy.io.loadmat(load_file)
        data = data['detections']
        detections.extend(data)
    return np.array(detections)


#   by ugly
def project_3d(u, v, cameraMatrix, distCoeffs, Rt):
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]
    k1 = distCoeffs[0]
    k2 = distCoeffs[1]
    k3 = distCoeffs[4]
    p1 = distCoeffs[2]
    p2 = distCoeffs[3]
    x_two = (u - cx) / fx
    y_two = (v - cy) / fy
    def f1(x):
        x_one = float(x[0])
        y_one = float(x[1])
        r2 = x_one * x_one + y_one * y_one
        r4 = x_one * x_one * x_one * x_one + y_one * y_one * y_one * y_one
        r6 = x_one * x_one * x_one * x_one * x_one * x_one + y_one * y_one * y_one * y_one * y_one * y_one
        return [x_two - (x_one * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x_one * y_one + p2 * (r2 + 2 * x_one * x_one)),
                y_two - (y_one * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2 * y_one * y_one) + 2 * p2 * x_one * y_one)]
    [x_one, y_one] = fsolve(f1, [0, 0])
    def f2(x):
        X = float(x[0])
        Y = float(x[1])
        return [x_one - ((Rt[0][0] * X + Rt[0][1] * Y + Rt[0][3]) / (Rt[2][0] * X + Rt[2][1] * Y + Rt[2][3])),
                y_one - ((Rt[1][0] * X + Rt[1][1] * Y + Rt[1][3]) / (Rt[2][0] * X + Rt[2][1] * Y + Rt[2][3]))]
    [world_x, world_y] = fsolve(f2, [0, 0])
    return world_x, world_y


def main():
    startFrame = 0
    endFrame = 1

    cmtx = np.loadtxt(matrix_save + 'intrinsics.txt')
    dist = np.loadtxt(matrix_save + 'distCoeffs.txt')
    Rt = []
    for i in range(1, cam_num + 1):
        Rt_temp = np.loadtxt(matrix_save + 'Rt' + str(i) + '.txt')
        Rt.append(Rt_temp)

    detections = load_detection(cam_num)
    for current_frame in range(startFrame, endFrame):
        detection = np.array([detections[i, [0, 7, 8]] for i in range(len(detections)) if detections[i, 1] == (current_frame + 1)])
        for i in range(0, len(detection)):
            # plot feet_x, feet_y into 3D location
            x, y = project_3d(detection[i, 1], detection[i, 2], cmtx, dist, Rt[int(detection[i, 0]) - 1])
            detection[i, 1] = x
            detection[i, 2] = y
        plt.plot(detection[:, 1], detection[:, 2], 'ro')
        plt.show()

if __name__ == '__main__':
    main()
