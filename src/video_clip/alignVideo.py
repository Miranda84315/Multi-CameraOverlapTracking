import cv2
import os

video_root = 'D:/Code/MultiCamOverlap/dataset/videos_old/Player'
save_root = 'D:/Code/MultiCamOverlap/dataset/videos/Player'
align_frame = [11, 14, 14, 0]

height = 1080
width = 1920
fps = 15
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for player in range(52, 53):
    for track in range(1, 9):
        if player < 10:
            video_filename = video_root + '0' + str(player) + '/track' + str(track)
            save_filename = save_root + '0' + str(player) + '/track' + str(track) 
        else:
            video_filename = video_root + str(player) + '/track' + str(track)
            save_filename = save_root + str(player) + '/track' + str(track)

        for icam in range(0, 4):
            filename = video_filename + '/cam' + str(icam + 1) + '.avi'
            out_filename = save_filename + '/cam' + str(icam + 1) + '.avi'
            print(out_filename)
            cap = cv2.VideoCapture(filename)
            out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))
            cap.set(1, align_frame[icam])
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, frame_num - align_frame[icam]):
                ret, frame = cap.read()
                cv2.imshow("video", frame)
                cv2.waitKey(1)
                out.write(frame)
            out.release()

        cv2.destroyAllWindows()
