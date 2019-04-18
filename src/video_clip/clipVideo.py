import cv2
import os
video_root = 'D:/Data/0317/'
video_filename = [
    'XVR_ch1_main_20190317083958_20190317100000.avi', 
    'XVR_ch2_main_20190317083958_20190317100000.avi',
    'XVR_ch3_main_20190317083958_20190317100000.avi',
    'XVR_ch4_main_20190317083958_20190317100000.avi']
save_root = 'D:/Data/dataset/Player05/track'
fps = 15
frame_start = [28546, 30191, 31384, 32814, 50704, 51783, 52176, 52530]
frame_end = [28753, 30580, 31832, 32971, 51500, 51970, 52373, 52684]

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
