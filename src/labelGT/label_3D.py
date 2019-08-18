import os
import cv2
import scipy.io
import numpy as np
import time
import pandas as pd
from scipy.optimize import fsolve
from argparse import ArgumentParser
'''
PlayerX(1~52) - trackY(1~8)
要記得改 line 12 ~ 13 的 PlayerX 以及 line14 的 trackY
還有matrix_save 的日期
'''
parser = ArgumentParser(description='show ui')
parser.add_argument('--track', type=int, required=True, help='Input cam number')
parser.add_argument('--player', type=int, required=True, help='Input cam number')
parser.add_argument('--day', type=str, required=True, help='Input cam number')

args = parser.parse_args()

track_num = str(args.track) + '/'
if args.player < 10:
    player = 'Player0' + str(args.player) + '/track'
else:
    player = 'Player' + str(args.player) + '/track'

save_dir = 'D:/Code/MultiCamOverlap/dataset/ground_truth/' + player
matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/' + args.day + '/information/'

save_root = save_dir + track_num


global refPt


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_coordinate(event, x, y, flags, param):
    global drawing, img, ix, iy, refPt
    drawing = False
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        refPt.append([ix, iy, x, y])


def load_mat(filename):
    data = scipy.io.loadmat(filename)
    data = data['gt']
    return np.array(data)


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

    cam_num = 4
    Rt = []
    cmtx = []
    dist = []
    cmtx = np.loadtxt(matrix_save + 'intrinsics.txt')
    dist = np.loadtxt(matrix_save + 'distCoeffs.txt')

    for i in range(1, cam_num + 1):
        Rt_temp = np.loadtxt(matrix_save + 'Rt' + str(i) + '.txt')
        Rt.append(Rt_temp)

    for i in range(1, 5):
        Rt_temp = np.loadtxt(matrix_save + 'Rt' + str(i) + '.txt')
        Rt.append(Rt_temp)

    gt = load_mat(save_root + 'gt_data.mat')
    end_sequence = max(gt[:, 0])
    gt_all = np.zeros((5 * end_sequence, 4))
    for frame in range(1, end_sequence + 1):
        for id_num in range(1, 6):
            gt_all[(id_num - 1) * end_sequence + frame - 1, 0] = frame
            gt_all[(id_num - 1) * end_sequence + frame - 1, 1] = id_num
            gt_temp = np.array([gt[i, [1, 3, 4, 5, 6]] for i in range(len(gt)) if gt[i, 0] == frame and gt[i, 2] == id_num])
            count = 0
            for icam, left, top, w, h in gt_temp:
                feet_x = int(left + (w/2))
                feet_y = int(top + h)
                x, y = project_3d(feet_x, feet_y, cmtx, dist, Rt[icam - 1])
                if x > 0 and y > 0:
                    gt_all[(id_num - 1) * end_sequence + frame - 1, 2] += x
                    gt_all[(id_num - 1) * end_sequence + frame - 1, 3] += y
                    count += 1
            gt_all[(id_num - 1) * end_sequence + frame - 1, 2] /= count
            gt_all[(id_num - 1) * end_sequence + frame - 1, 3] /= count
    scipy.io.savemat(save_root + 'gt_data_3D.mat', mdict={'gt_3D': gt_all})
    print(save_root + 'gt_data_3D.mat')


if __name__ == '__main__':
    main()