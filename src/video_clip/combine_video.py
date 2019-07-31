import cv2
import numpy as np
import scipy.io
import os.path


def main():
    startFrame = 0
    frame_total = [152, 117, 153, 180, 114, 201, 158, 364]

    for num in range(1, 9):
        track_num = str(num) + '/'
        video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player52/track'
        experiment_dir = 'D:/Code/MultiCamOverlap/temp/Player52/'

        experiment_root = experiment_dir
        video_root = video_dir + track_num

        cam_num = 4
        width = 1920
        height = 1080
        fps = 15

        endFrame = frame_total[num - 1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = experiment_root + 'camera_result' + str(num) + '.avi'
        print(out_filename)
        height = 1080
        width = 1920
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

        filename = [video_root + 'cam1.avi', video_root + 'cam2.avi', video_root + 'cam3.avi', video_root + 'cam4.avi']

        cap1 = cv2.VideoCapture(filename[0])
        cap2 = cv2.VideoCapture(filename[1])
        cap3 = cv2.VideoCapture(filename[2])
        cap4 = cv2.VideoCapture(filename[3])
        cap = [cap1, cap2, cap3, cap4]

        for current_frame in range(startFrame, endFrame):
            img_temp = []
            for icam in range(1, cam_num + 1):
                ret, frame_img = cap[icam-1].read()
                img_temp.append(frame_img)

            img_top = np.concatenate((img_temp[0], img_temp[1]), axis=0)
            img_bottom = np.concatenate((img_temp[2], img_temp[3]), axis=0)
            img = np.concatenate((img_top, img_bottom), axis=1)
            img = cv2.resize(img, (1920, 1080))

            print('frame = ' + str(current_frame) + ' / ' + str(endFrame))
            out.write(img)

        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
