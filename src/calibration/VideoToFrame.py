import cv2
import os
'''
    This is for load video and save the frame
'''

dataset_root = 'D:/Code/MultiCamOverlap/dataset/calibration/'
save_root = 'D:/Code/MultiCamOverlap/dataset/calibration/'
cam_num = 4

for icam in range(1, cam_num + 1):
    video_name = dataset_root + 'cam' + str(icam) + '.avi'
    print('===== Load vodeo: ' + str(video_name) + ' =====')
    cap = cv2.VideoCapture(video_name)
