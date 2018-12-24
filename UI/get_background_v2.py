import cv2
import numpy as np
'''
This is for terrace dataset
'''


experiment_root = 'D:/Code/MultiCamOverlap/experiments/demo/'
visual_root = 'D:/Code/MultiCamOverlap/UI/data/'
video_root = 'D:/Code/MultiCamOverlap/dataset/videos/No1/'

start_time = [1, 1, 1, 1]
NumFrames = [225, 225, 225, 225]
PartFrames = [[225, 225, 225, 225]]
cam_num = 4


start_sequence = 0
end_sequence = 225


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 1):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def get_frame(icam, frame_num):
    #   only show video
    part_cam, part_frame = calucate_part(icam, frame_num)
    filename = video_root + 'cam' + str(icam) + '.avi'
    cap = cv2.VideoCapture(filename)
    cap.set(1, part_frame)
    ret, frame_img = cap.read()
    cv2.imshow("video", frame_img)
    cv2.waitKey(1)
    path_name = visual_root + 'background' + str(icam) + '.jpg'
    cv2.imwrite(path_name, frame_img)
    cap.release()


def get_map(icam):
    part_cam, part_frame = calucate_part(icam, 0)
    filename = video_root + 'cam' + str(icam) + '.avi'
    cap = cv2.VideoCapture(filename)
    ret, frame_img = cap.read()
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    blank_image = np.full((height, width, 3), 255)
    path_name = visual_root + 'map.jpg'
    cv2.imwrite(path_name, blank_image)


def main():
    frame = [0, 0, 0, 0]
    for i in range(1, cam_num + 1):
        get_frame(i, frame[i - 1])

    get_map(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
