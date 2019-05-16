import os
import cv2
import scipy.io
import numpy as np
import time
import pandas as pd
from scipy.optimize import fsolve
'''
PlayerX(1~52) - trackY(1~8)
要記得改 line 12 ~ 13 的 PlayerX 以及 line14 的 trackY
'''
video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player05/track'
save_dir = 'D:/Code/MultiCamOverlap/dataset/ground_truth/Player05/track'
track_num = '2/'

video_root = video_dir + track_num
save_root = save_dir + track_num
global refPt


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_frame_number(cam_num, video_root):
    frame = []
    for icam in range(1, cam_num + 1):
        filename = video_root + 'cam' + str(icam) + '.avi'
        cap = cv2.VideoCapture(filename)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frame num = ', frame_num)
        frame.append(frame_num)
    return min(frame)


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
    start_sequence = 0
    end_sequence = get_frame_number(cam_num, video_root)
    createFolder(video_root)
    createFolder(save_root)

    filename = [video_root + 'cam1.avi', video_root + 'cam2.avi', video_root + 'cam3.avi', video_root + 'cam4.avi']

    cap1 = cv2.VideoCapture(filename[0])
    cap2 = cv2.VideoCapture(filename[1])
    cap3 = cv2.VideoCapture(filename[2])
    cap4 = cv2.VideoCapture(filename[3])
    cap = [cap1, cap2, cap3, cap4]

    global refPt, img
    start = time.time()
    gt = []

    for id_num in range(1, 6):
        for icam in range(1, cam_num + 1):
            gt_cam = []
            for frame_num in range(start_sequence, end_sequence, 5):
                print('id = ', id_num, ', / cam = ', icam, ', / frame = ', frame_num)
                cap[icam-1].set(1, frame_num)
                ret, img = cap[icam-1].read()
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', get_coordinate)
                refPt = []
                while(1):
                    cv2.imshow('image', img)
                    k = cv2.waitKey(33)
                    if k == 32:     # space to stop
                        break
                if len(refPt) >= 1:
                    refPt = refPt[-1]
                    left = refPt[0]
                    top = refPt[1]
                    width = refPt[2] - refPt[0]
                    height = refPt[3] - refPt[1]
                    gt_temp = [icam, id_num, frame_num + 1, left, top, width, height]
                    gt_cam.append(gt_temp)
                print(gt_temp)
                cv2.destroyAllWindows()
            gt_cam = np.array(gt_cam)
            gt_cam = gt_cam.reshape((len(gt_cam), 7))
            gt_new = pd.DataFrame(gt_cam, columns=['icam', 'id', 'frame', 'left', 'top', 'w', 'h'])
            gt_new = gt_new.set_index('frame')
            gt_new = gt_new.reindex(np.arange(gt_new.index.min(), gt_new.index.max()+1))
            gt_new = gt_new.interpolate()
            gt_new = gt_new.reset_index()
            gt_new = np.array(gt_new)
            gt.extend(gt_new)

    gt = np.array(gt, dtype=int)
    scipy.io.savemat(save_root + 'gt_data.mat', mdict={'gt': gt})
    end = time.time()
    elapsed = end - start
    print('\n\ntime: ', elapsed)
    cv2.destroyAllWindows()

    have3D = False
    if have3D:
        matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/Player05/information/'
        cmtx = np.loadtxt(matrix_save + 'intrinsics.txt')
        dist = np.loadtxt(matrix_save + 'distCoeffs.txt')
        Rt = []

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
                for icam, left, top, w, h in gt_temp:
                    feet_x = int(left + (w/2))
                    feet_y = int(top + h)
                    x, y = project_3d(feet_x, feet_y, cmtx, dist, Rt[icam - 1])
                    gt_all[(id_num - 1) * end_sequence + frame - 1, 2] += x
                    gt_all[(id_num - 1) * end_sequence + frame - 1, 3] += y
                gt_all[(id_num - 1) * end_sequence + frame - 1, 2] /= len(gt_temp)
                gt_all[(id_num - 1) * end_sequence + frame - 1, 3] /= len(gt_temp)
        scipy.io.savemat(save_root + 'gt_data_3D.mat', mdict={'gt_3D': gt_all})


if __name__ == '__main__':
    main()