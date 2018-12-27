import cv2
import numpy as np
import scipy.io
from argparse import ArgumentParser
'''
python ./nms.py --icam 1
'''

file_name = 'D:/Code/TrackingSystem/dataset/detections/top1/camera'
videos_root = 'D:/Code/DukeMTMC/videos/camera'
save_root = 'D:/Code/TrackingSystem/dataset/detections/top1/index'
start_time = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
PartFrames = [[38370, 38370, 38400, 38670, 38370, 38400, 38790, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38640, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38460, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38610, 38370],
              [38370, 38370, 38370, 38670, 38370, 38400, 38760, 38370],
              [38370, 38370, 38370, 38700, 38370, 38400, 38760, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38790, 38370],
              [38370, 38370, 38370, 38670, 38370, 38370, 38490, 38370],
              [38370, 38370, 38370, 38670, 38370, 37350, 28380, 38370],
              [14250, 15390, 10020, 26790, 21060, 0, 0, 7890]]


parser = ArgumentParser(description='objection detection NMS')
parser.add_argument('--icam', required=True, help='Input cam number')


def load_mat(icam):
    data = scipy.io.loadmat(file_name + str(icam) + '.mat')
    detections = data['detections']
    return detections


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 10):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def cal_localtime(icam, frame_num):
    # get the real locat time
    return frame_num - start_time[icam - 1] + 1


def find_index(icam, frame_local, detections):
    index = [i for i in range(len(detections)) if detections[i, 1] == frame_local]
    return index


def draw_nms_img(img, icam, keep, detections):
    for ind in keep:
        left_x = int(detections[ind, 2])
        left_y = int(detections[ind, 3])
        right_x = int(detections[ind, 2] + detections[ind, 4])
        right_y = int(detections[ind, 3] + detections[ind, 5])
        img = cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (255, 0, 255), 3)
    return img


def nms(detections, ind, thresh):

    x1 = detections[ind, 2]
    y1 = detections[ind, 3]
    x2 = detections[ind, 2] + detections[ind, 4]
    y2 = detections[ind, 3] + detections[ind, 5]
    scores = detections[ind, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep_temp = []
    while order.size > 0:
        i = order[0]
        keep_temp.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    keep = []
    for k in keep_temp:
        keep.append(ind[k])

    return keep


def main():
    args = parser.parse_args()
    startFrame_global = 127720
    endFrame_global = 187540
    icam = int(args.icam)
    detections = load_mat(icam)
    #part_cam_previous = -1
    keep_ind = []

    for frame in range(startFrame_global, endFrame_global):
        print(frame)
        frame_local = cal_localtime(icam, frame)
        # part_cam, part_frame = calucate_part(icam, frame_local)
        # if frame == startFrame_global:
        #     filename = videos_root + str(icam) + '/0000' + str(
        #         part_cam) + '.MTS'
        #     cap = cv2.VideoCapture(filename)
        #     cap.set(1, part_frame)
        #     part_cam_previous = part_cam
        # if part_cam != part_cam_previous:
        #     filename = videos_root + str(icam) + '/0000' + str(
        #         part_cam) + '.MTS'
        #     cap = cv2.VideoCapture(filename)
        # part_cam_previous = part_cam
        # ret, frame_img = cap.read()
        index = find_index(icam, frame_local, detections)
        keep = nms(detections, index, 0.3)
        print(keep)
        keep_ind.extend(keep)
    print(keep_ind)
    np.savetxt(save_root + str(icam) + '.txt', keep_ind, fmt='%d')


#    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
