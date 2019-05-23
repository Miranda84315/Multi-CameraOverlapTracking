import cv2
import numpy as np
import scipy.io
import os.path
from heatmappy import Heatmapper
from PIL import Image
'''
This is use for save tracking result video
And save in 
video/camera1_result.avi
video/camera2_result.avi
...
video/camera8_result.avi
'''

track_num = '2/'
video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player05/track'
experiment_dir = 'D:/Code/MultiCamOverlap/experiments/Player05/track'

experiment_root = experiment_dir + track_num
visual_root = 'D:/Code/MultiCamOverlap/UI/data/'
video_root = video_dir + track_num

start_time = [1, 1, 1, 1]
cam_num = 4
threshold_durations = 10 * 25
board_A = 10
board_B = 8
width = 1920
height = 1080
fps = 15


def load_mat():
    load_file = experiment_root + 'L2-trajectories/L2_cam.mat'
    trajectory = scipy.io.loadmat(load_file)
    data = trajectory['fileOutput']
    return data


def random_color(number_people):
    color = np.zeros((number_people + 1, 3))
    for i in range(0, number_people + 1):
        color[i] = list(np.random.choice(range(256), size=3))
    return color


def cal_localtime(icam, frame_num):
    # get the real locat time
    start_sequence = 0
    return start_sequence + frame_num - start_time[icam - 1] + 1


def draw_bb(img, icam, data):
    for detection in data:
        color_id = tuple(color[int(detection[1])])
        index = icam * 4
        left_x = int(detection[index])
        left_y = int(detection[index + 1])
        right_x = int(detection[index] + detection[index + 2])
        right_y = int(detection[index + 1] + detection[index + 3])
        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color_id, 3)
        cv2.rectangle(img, (left_x - 1, left_y - 20), (right_x + 1, left_y), color_id, -1)
        cv2.putText(img, str(int(detection[1])), (left_x, left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def draw_traj(img, frame, data):
    for detection in data:
        color_id = tuple(color[int(detection[1])])
        px = int(detection[2])
        py = int(detection[3])
        cv2.circle(img, (px, py), 30, color_id, -1)
    return img


def main():
    startFrame = 0
    #endFrame = 390

    global fileOutput
    global color
    fileOutput = load_mat()
    color = random_color(len(set(fileOutput[:, 1])))
    endFrame = int(max(fileOutput[:, 0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_filename = experiment_root + 'camera_result.avi'
    height = 1080
    width = 1920
    out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
    traj_filename = experiment_root + 'trajectory_result.avi'
    traj = cv2.VideoWriter(traj_filename, fourcc, fps, (800, 700))

    filename = [video_root + 'cam1.avi', video_root + 'cam2.avi', video_root + 'cam3.avi', video_root + 'cam4.avi']

    cap1 = cv2.VideoCapture(filename[0])
    cap2 = cv2.VideoCapture(filename[1])
    cap3 = cv2.VideoCapture(filename[2])
    cap4 = cv2.VideoCapture(filename[3])
    cap = [cap1, cap2, cap3, cap4]

    for current_frame in range(startFrame, endFrame):
        ind = [fileOutput[i, :] for i in range(0, len(fileOutput)) if fileOutput[i, 0] == current_frame + 1]
        img_temp = []
        for icam in range(1, cam_num + 1):
            ret, frame_img = cap[icam-1].read()
            frame_img = draw_bb(frame_img, icam, ind)
            img_temp.append(frame_img)

        img_top = np.concatenate((img_temp[0], img_temp[1]), axis=0)
        img_bottom = np.concatenate((img_temp[2], img_temp[3]), axis=0)
        img = np.concatenate((img_top, img_bottom), axis=1)
        img = cv2.resize(img, (1920, 1080))

        cv2.putText(img, str(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("video2", img)
        trajectory = cv2.imread('data/BasketballCourt.png')
        trajectory_img = cv2.resize(draw_traj(trajectory, current_frame, ind), (800, 700))
        cv2.imshow("video", trajectory_img)
        cv2.waitKey(1)
        print('frame = ' + str(current_frame) + ' / ' + str(endFrame))
        out.write(img)
        traj.write(trajectory_img)

    out.release()
    traj.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
