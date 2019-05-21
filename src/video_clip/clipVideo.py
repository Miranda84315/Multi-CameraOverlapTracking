import cv2
import os
video_root = 'D:/Data/0425/'
video_filename = [
    'XVR_ch1_main_20190425110000_20190425115800.avi',
    'XVR_ch2_main_20190425110000_20190425115800.avi',
    'XVR_ch3_main_20190425110000_20190425115800.avi',
    'XVR_ch4_main_20190425110001_20190425115800.avi']
video_filename2 = [
    'XVR_ch1_main_20190315140000_20190315142353.avi',
    'XVR_ch2_main_20190315140000_20190315142353.avi',
    'XVR_ch3_main_20190315140000_20190315142353.avi',
    'XVR_ch4_main_20190315140001_20190315142353.avi']
save_root = 'D:/Data/dataset/Player47/track'
save_dir = 'D:/Data/dataset/Player47'
fps = 15
frame_start = [25340, 26311, 27375, 28462, 50144, 50619, 50919, 51350]
frame_end = [25529, 26552, 27624, 28742, 50505, 50778, 51155, 51612]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for track in range(0, 8):
    path = save_root + str(track + 1)
    if not os.path.exists(path):
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
        '''
        cap.set(1, frame_start[clip_num] - 1)
        for i in range(frame_start[clip_num] - 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # print(i)
            ret, frame = cap.read()
            cv2.imshow("video", frame)
            cv2.waitKey(1)
            out.write(frame)
        filename = video_root + video_filename2[icam]
        cap = cv2.VideoCapture(filename)
        for i in range(0, frame_end[clip_num]):
            ret, frame = cap.read()
            cv2.imshow("video", frame)
            cv2.waitKey(1)
            out.write(frame)
        '''
        out.release()

cv2.destroyAllWindows()
