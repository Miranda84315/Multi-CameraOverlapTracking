import cv2
import os
video_root = 'D:/Data/0317/'
video_filename = [
    'XVR_ch1_main_20190317160000_20190317162500.avi',
    'XVR_ch2_main_20190317160000_20190317162500.avi',
    'XVR_ch3_main_20190317160000_20190317162500.avi',
    'XVR_ch4_main_20190317160000_20190317162500.avi']
save_root = 'D:/Data/dataset/Player10/track'
fps = 15
frame_start = [12791, 13925, 14940, 15868, 20078, 20401, 20764, 21228]
frame_end = [13109, 14277, 15250, 16061, 20236, 20559, 20999, 21452]

for track in range(0, 8):
    path = save_root + str(track + 1)
    os.mkdir(path)


for icam in range(0, 4):
    filename = video_root + video_filename[icam]
    cap = cv2.VideoCapture(filename)
    for clip_num in range(0, 8):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = save_root + str(clip_num + 1) + '/cam' + str(icam + 1) + '.avi'
        print('======================================')
        print(out_filename)
        height = 1080
        width = 1920
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
        cap.set(1, frame_start[clip_num] - 1)
        for i in range(frame_start[clip_num] - 1, frame_end[clip_num]):
            # print(i)
            ret, frame = cap.read()
            cv2.imshow("video", frame)
            cv2.waitKey(1)
            out.write(frame)
        out.release()

cv2.destroyAllWindows()
