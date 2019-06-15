import numpy as np
import os
from matplotlib.path import Path
import scipy.io

calibration_dir = [
    'D:/Code/MultiCamOverlap/dataset/calibration/0315/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0315_02/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0315_03/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0315_04/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317_08/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317_09/information/'
]
track_num = ['1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/']


def inROI(icam, x, y):
    return p[icam - 1].contains_points([(x, y)])[0]


def load_detection(filename):
    data = scipy.io.loadmat(filename)
    data = data['detections']
    return np.array(data)


def load_pose(filename):
    data = scipy.io.loadmat(filename)
    data = data['poses']
    return np.array(data)


def main(player, track):
    global video_dir, save_dir
    cam_num = 4
    path_video = video_dir[player] + track_num[track]
    path_save = save_dir[player] + track_num[track]
    for icam in range(1, cam_num + 1):
        video_file = path_video + 'cam' + str(icam) + '.avi'
        out_dir = path_save
        cmd = 'C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe video_demo.py --video ' + video_file + ' --outdir ' + out_dir + ' --icam ' + str(icam) + ' --sp'
        print(cmd)
        os.system(cmd)
        mat_save = path_save + 'cam' + str(icam) + '.mat'
        pose_save = path_save + 'cam' + str(icam) + '_pose.mat'
        detections = load_detection(mat_save)
        poses = load_pose(pose_save)
        roi_ind = []
        for i in range(0, len(detections)):
            feet_x = detections[i, 7]
            feet_y = detections[i, 8]
            roi = inROI(icam, int(feet_x), int(feet_y))
            if roi:
                roi_ind.append(i)
        #print(roi_ind)
        new_detections = detections[roi_ind, :]
        new_poses = poses[roi_ind, :]
        scipy.io.savemat(path_save + 'cam' + str(icam) + '.mat', mdict={'detections': new_detections})
        scipy.io.savemat(path_save + 'cam' + str(icam) + '_pose.mat', mdict={'poses': new_poses})


if __name__ == '__main__':
    global video_dir
    global save_dir
    global p
    video_root = 'D:/Code/MultiCamOverlap/dataset/videos/Player0'
    save_root = 'D:/Code/MultiCamOverlap/dataset/alpha_pose/Player0'
    video_dir = [
        video_root + str(1) + '/track',
        video_root + str(2) + '/track',
        video_root + str(3) + '/track',
        video_root + str(4) + '/track',
        video_root + str(5) + '/track',
        video_root + str(6) + '/track',
        video_root + str(7) + '/track',
        video_root + str(8) + '/track',
        video_root + str(9) + '/track'
    ]
    save_dir = [
        save_root + str(1) + '/track',
        save_root + str(2) + '/track',
        save_root + str(3) + '/track',
        save_root + str(4) + '/track',
        save_root + str(5) + '/track',
        save_root + str(6) + '/track',
        save_root + str(7) + '/track',
        save_root + str(8) + '/track',
        save_root + str(9) + '/track',
    ]
    for player in range(0, 9):
        print('player: ', video_dir[player])
        roi_filename = calibration_dir[player] + 'ROI.npy'
        p = np.load(roi_filename)
        for track in range(0, 8):
            print('track : ', track + 1)
            main(player, track)
            system_cmd = 'C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe D:/Code/MultiCamOverlap/src/detection/combine_detection.py --track ' + track_num[track] + ' --calibration ' + calibration_dir[player] + ' --save ' + save_dir[player]
            print(system_cmd)
            os.system(system_cmd)