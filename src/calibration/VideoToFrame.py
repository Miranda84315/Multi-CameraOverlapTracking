import cv2

'''
    This is for load video and save the frame
    1. To get calibration image, one frame per camera:
        only get the first frame of video
    2. To get calibration image, each frame from video:
        get all frame from video
'''
get_all = True
get_one = False

dataset_root = 'D:/Code/MultiCamOverlap/dataset/calibration/'
save_oneframe = 'D:/Code/MultiCamOverlap/dataset/calibration/'
save_allframe = 'D:/Code/MultiCamOverlap/dataset/calibration/cam'
cam_num = 4

'''
    1. To Get calibration image, one frame per camera
'''
if get_one:
    for icam in range(1, cam_num + 1):
        video_name = dataset_root + 'cam' + str(icam) + '.avi'
        print('===== Load vodeo: ' + str(video_name) + ' =====')
        cap = cv2.VideoCapture(video_name)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        file_name = save_oneframe + 'cam' + str(icam) + '.jpg'
        print('save: ' + file_name)
        ret, frame = cap.read()
        cv2.imwrite(file_name, frame)

'''
    1. To get calibration image, each frame from video
'''
if get_all:
    for icam in range(1, cam_num + 1):
        video_name = dataset_root + 'cam' + str(icam) + '.avi'
        print('===== Load vodeo: ' + str(video_name) + ' =====')
        cap = cv2.VideoCapture(video_name)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(1, frame_num + 1):
            file_name = save_allframe + str(icam) + '/' + str(i) + '.jpg'
            print('save: ' + file_name)
            ret, frame = cap.read()
            cv2.imwrite(file_name, frame)
