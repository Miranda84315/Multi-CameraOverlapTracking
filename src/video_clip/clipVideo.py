import cv2
import os
video_root = 'D:/Data/0318/'
video_filename = [
    'XVR_ch1_main_20190318110001_20190318114828.avi',
    'XVR_ch2_main_20190318110001_20190318114828.avi',
    'XVR_ch3_main_20190318110000_20190318114828.avi',
    'XVR_ch4_main_20190318110000_20190318114828.avi']
save_root = 'D:/Data/dataset/Player14/track'
save_dir = 'D:/Data/dataset/Player14'
fps = 15
frame_start = [23483, 24994, 26200, 27654, 42736, 43405, 43930, 44705]
frame_end = [23733, 25284, 26393, 27929, 43036, 43641, 44261, 44888]

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
        out.release()

cv2.destroyAllWindows()
