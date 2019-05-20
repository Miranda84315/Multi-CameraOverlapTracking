import cv2
import os
video_root = 'D:/Data/0415/'
video_filename = [
    'XVR_ch1_main_20190415110000_20190415113822.avi',
    'XVR_ch2_main_20190415110001_20190415113822.avi',
    'XVR_ch3_main_20190415110000_20190415113822.avi',
    'XVR_ch4_main_20190415110000_20190415113822.avi']
video_filename2 = [
    'XVR_ch1_main_20190415110000_20190415113822.avi',
    'XVR_ch2_main_20190415110001_20190415113822.avi',
    'XVR_ch3_main_20190415110000_20190415113822.avi',
    'XVR_ch4_main_20190415110000_20190415113822.avi']
save_root = 'D:/Data/dataset/Player31/track'
save_dir = 'D:/Data/dataset/Player31'
fps = 15
frame_start = [36344, 37322, 38384, 39272, 3488, 3833, 4402, 4889]
frame_end = [36508, 37549, 38581, 39403, 3694, 4032, 4766, 5226]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for track in range(0, 8):
    path = save_root + str(track + 1)
    if not os.path.exists(path):
        os.mkdir(path)


for icam in range(0, 4):
    filename = video_root + video_filename[icam]
    cap = cv2.VideoCapture(filename)
    for clip_num in range(4, 8):
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
        # if end < start
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
