import cv2
import numpy as np
import scipy.io
import os.path
from heatmappy import Heatmapper
from PIL import Image
from argparse import ArgumentParser


parser = ArgumentParser(description='show ui')
parser.add_argument('--track', type=int, required=True, help='Input cam number')
parser.add_argument('--player', type=int, required=True, help='Input cam number')

args = parser.parse_args()

track_num = str(args.track) + '/'
if args.player < 10:
    player = 'Player0' + str(args.player) + '/track'
else:
    player = 'Player' + str(args.player) + '/track'
gt_dir = 'D:/Code/MultiCamOverlap/dataset/ground_truth/' + player
experiment_dir = 'D:/Code/MultiCamOverlap/experiments_alpha/' + player

#track_num = '6/'
#video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player05/track'
#experiment_dir = 'D:/Code/MultiCamOverlap/experiments/Player05/track'

experiment_root = experiment_dir + track_num
visual_root = 'D:/Code/MultiCamOverlap/UI/data/'
gt_root = gt_dir + track_num

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


def load_gt():
    load_file = gt_root + 'gt_data_3D.mat'
    trajectory = scipy.io.loadmat(load_file)
    data = trajectory['gt_3D']
    return data


def random_color(number_people):
    color = np.zeros((number_people + 1, 3))
    color[0] = [82, 82, 82]     # yellow
    color[1] = [119, 65, 110]   # purple
    color[2] = [213, 173, 71]   # blue
    color[3] = [111, 255, 80]   # green
    color[4] = [74, 38, 255]    # red
    color[5] = [7, 211, 252]
    for i in range(6, number_people + 1):
        color[i] = list(np.random.choice(range(256), size=3))
    return color


def cal_localtime(icam, frame_num):
    # get the real locat time
    start_sequence = 0
    return start_sequence + frame_num - start_time[icam - 1] + 1


def draw_traj_3D(img, frame, data, gt):
    for detection in data:
        color_id = tuple(color[int(detection[1])])
        px = int(detection[2])
        py = int(detection[3])
        cv2.circle(img, (px, py), 30, color_id, -1)
    for detection in gt:
        color_id = tuple(color[0])
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
    gtOutput = load_gt()
    color = random_color(len(set(fileOutput[:, 1])))
    endFrame = int(max(fileOutput[:, 0]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    traj_filename = experiment_root + 'trajectory_3D.avi'
    traj = cv2.VideoWriter(traj_filename, fourcc, fps, (800, 700))

    for current_frame in range(startFrame, endFrame):
        ind = [fileOutput[i, :] for i in range(0, len(fileOutput)) if fileOutput[i, 0] == current_frame + 1]
        ind_gt = [gtOutput[i, :] for i in range(0, len(gtOutput)) if gtOutput[i, 0] == current_frame + 1]
        trajectory = cv2.imread('D:/Code/MultiCamOverlap/UI/data/BasketballCourt.png')
        trajectory_img = cv2.resize(draw_traj_3D(trajectory, current_frame, ind, ind_gt), (800, 700))
        cv2.imshow("video", trajectory_img)
        cv2.waitKey(1)
        print('frame = ' + str(current_frame) + ' / ' + str(endFrame))
        traj.write(trajectory_img)

    traj.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
