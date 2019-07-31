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
    'D:/Code/MultiCamOverlap/dataset/calibration/0317_09/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0317_09/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0318/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0318/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0318/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0322_14/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0322_15/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0322/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0322/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0408/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0408/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0408/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412_22/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0412/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0415_29/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0415/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0415_31/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0419_32/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0419/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0419_34/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0419_35/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0421_36/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0421_37/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0421/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0421/information/',
    'D:/Code/MultiCamOverlap/dataset/calibration/0421/information/'
]
track_num = ['1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/']


def inROI(icam, x, y):
    return p[icam - 1].contains_points([(x, y)])[0]


def load_detection(filename):
    data = scipy.io.loadmat(filename)
    data = data['detections']
    return np.array(data)


def main(player, track):
    global video_dir, save_dir
    cam_num = 4
    path_video = video_dir[player] + track_num[track]
    path_save = save_dir[player] + track_num[track]
    for icam in range(1, cam_num + 1):
        video_file = path_video + 'cam' + str(icam) + '.avi'
        out_dir = path_save
        cmd = 'python run.py --model=cmu --resize=432x368 --video=' + video_file + ' --icam=' + str(icam) + ' --save_root=' + out_dir
        # print(cmd)
        # os.system(cmd)

        mat_save = path_save + 'cam' + str(icam) + '.mat'
        detections = load_detection(mat_save)
        roi_ind = []
        for i in range(0, len(detections)):
            feet_x = detections[i, 7]
            feet_y = detections[i, 8]
            roi = inROI(icam, int(feet_x), int(feet_y))
            if roi:
                roi_ind.append(i)
        # print(roi_ind)
        new_detections = detections[roi_ind, :]
        scipy.io.savemat(path_save + 'cam' + str(icam) + '.mat', mdict={'detections': new_detections})


if __name__ == '__main__':
    global video_dir
    global save_dir
    global p
    video_root = 'D:/Code/MultiCamOverlap/dataset/videos/Player'
    save_root = 'D:/Code/MultiCamOverlap/dataset/openpose/Player'
    video_dir = [
        video_root + '01/track',
        video_root + '02/track',
        video_root + '03/track',
        video_root + '04/track',
        video_root + '05/track',
        video_root + '06/track',
        video_root + '07/track',
        video_root + '08/track',
        video_root + '09/track',
        video_root + '10/track',
        video_root + '11/track',
        video_root + '12/track',
        video_root + '13/track',
        video_root + '14/track',
        video_root + '15/track',
        video_root + '16/track',
        video_root + '17/track',
        video_root + '19/track',
        video_root + '20/track',
        video_root + '21/track',
        video_root + '22/track',
        video_root + '23/track',
        video_root + '24/track',
        video_root + '25/track',
        video_root + '26/track',
        video_root + '27/track',
        video_root + '28/track',
        video_root + '29/track',
        video_root + '30/track',
        video_root + '31/track',
        video_root + '32/track',
        video_root + '33/track',
        video_root + '34/track',
        video_root + '35/track',
        video_root + '36/track',
        video_root + '37/track',
        video_root + '38/track',
        video_root + '39/track',
        video_root + '40/track'
    ]
    save_dir = [
        save_root + '01/track',
        save_root + '02/track',
        save_root + '03/track',
        save_root + '04/track',
        save_root + '05/track',
        save_root + '06/track',
        save_root + '07/track',
        save_root + '08/track',
        save_root + '09/track',
        save_root + '10/track',
        save_root + '11/track',
        save_root + '12/track',
        save_root + '13/track',
        save_root + '14/track',
        save_root + '15/track',
        save_root + '16/track',
        save_root + '17/track',
        save_root + '19/track',
        save_root + '20/track',
        save_root + '21/track',
        save_root + '22/track',
        save_root + '23/track',
        save_root + '24/track',
        save_root + '25/track',
        save_root + '26/track',
        save_root + '27/track',
        save_root + '28/track',
        save_root + '29/track',
        save_root + '30/track',
        save_root + '31/track',
        save_root + '32/track',
        save_root + '33/track',
        save_root + '34/track',
        save_root + '35/track',
        save_root + '36/track',
        save_root + '37/track',
        save_root + '38/track',
        save_root + '39/track',
        save_root + '40/track'
    ]
    for player in range(0, 39):
        print('player: ', video_dir[player])
        roi_filename = calibration_dir[player] + 'ROI.npy'
        p = np.load(roi_filename)
        for track in range(0, 8):
            print('track : ', track + 1)
            main(player, track)
            system_cmd = 'C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe D:/Code/MultiCamOverlap/src/detection/combine_detection.py --track ' + track_num[track] + ' --calibration ' + calibration_dir[player] + ' --save ' + save_dir[player]
            print(system_cmd)
            os.system(system_cmd)
