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


video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/' + player

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


def load_mat(filename):
    data = scipy.io.loadmat(filename)
    data = data['gt']
    return np.array(data)


def main():
    cam_num = 4
    start_sequence = 0
    end_sequence = get_frame_number(cam_num, video_root)

    gt = load_mat(save_root + 'gt_data.mat')
    gt_reconstruct = []

    ind = range(start_sequence + 1, end_sequence + 1 , 5)
    for id_num in range(1, 6):
        for icam in range(1, cam_num + 1):
            gt_cam = np.array([gt[i, :] for i in range(len(gt)) if gt[i, 0] in ind and gt[i, 1] == icam and gt[i, 2] == id_num])
            for i in range(0, len(gt_cam)):
                if gt_cam[i, 5] < 0 :
                    gt_cam[i, 3] = gt_cam[i, 3] + gt_cam[i, 5]
                    gt_cam[i, 5] = abs(gt_cam[i, 5])
                if gt_cam[i, 6] < 0:
                    gt_cam[i, 4] = gt_cam[i, 4] + gt_cam[i, 6]
                    gt_cam[i, 6] = abs(gt_cam[i, 6])
            gt_new = pd.DataFrame(gt_cam, columns=['frame', 'icam', 'id', 'left', 'top', 'w', 'h'])
            gt_new = gt_new.set_index('frame')
            gt_new = gt_new.reindex(np.arange(gt_new.index.min(), gt_new.index.max()+1))
            gt_new = gt_new.interpolate()
            gt_new = gt_new.reset_index()
            gt_new = np.array(gt_new)
            gt_reconstruct.extend(gt_new)

    gt = np.array(gt_reconstruct, dtype=int)
    scipy.io.savemat(save_root + 'gt_data_rec.mat', mdict={'gt': gt})


if __name__ == '__main__':
    main()