import numpy as np
import dlib
import math
import os
import cv2


detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor()

experiment_root = 'D:/Code/MultiCamOverlap/experiments/demo/'
visual_root = 'D:/Code/MultiCamOverlap/UI/data/'
video_root = 'D:/Code/MultiCamOverlap/dataset/videos/No1/'

start_time = [1, 1, 1, 1]
NumFrames = [225, 225, 225, 225]
PartFrames = [[225, 225, 225, 225]]
cam_num = 4
threshold_durations = 10 * 25
board_A = 10
board_B = 8
width = 1920
height = 1080
fps = 15


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def get_img(iCam, frame):
    part_cam, part_frame = calucate_part(iCam, frame)
    filename = 'D:/Code/DukeMTMC/videos/camera' + str(iCam) + '/0000' + str(part_cam) + '.MTS'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    _, img = cap.read()
    return img


def get_facepoint(img):
    img2 = img
    faces = detector(img, 1)

    if len(faces) > 0:
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
    return img2


def main():
    icam = 1
    startFrame = 0
    endFrame = 50
    cam_num = 4

    for icam in range(2, cam_num + 1):
        for current_frame in range(startFrame, endFrame):
            part_cam, part_frame = calucate_part(icam, current_frame)
            if current_frame == startFrame:
                filename = video_root + 'cam' + str(icam) + '.avi'
                cap = cv2.VideoCapture(filename)
            ret, frame_img = cap.read()
            img = get_facepoint(frame_img)
            cv2.imshow("video2", img)
            cv2.waitKey(1)
            print('frame = ', current_frame)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
