import os
import cv2
import scipy.io
import numpy as np
import time
import pandas as pd

video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player05/track'
save_dir = 'D:/Code/MultiCamOverlap/dataset/ground_truth/Player05/track'
track_num = '1/'

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


if __name__ == '__main__':
    main()