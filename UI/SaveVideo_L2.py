import cv2
import numpy as np
import scipy.io
import os.path
from heatmappy import Heatmapper
from PIL import Image
'''
This is use for save tracking result video
And save in 
video/camera1_result.avi
video/camera2_result.avi
...
video/camera8_result.avi
'''

experiment_root = 'D:/Code/MultiCamOverlap/experiments/demo_test/'
visual_root = 'D:/Code/MultiCamOverlap/UI/data/'
video_root = 'D:/Code/MultiCamOverlap/dataset/videos/No3/'

start_time = [1, 1, 1, 1]
NumFrames = [810, 810, 810, 810]
PartFrames = [[810, 810, 810, 810]]
cam_num = 4
threshold_durations = 10 * 25
board_A = 10
board_B = 8
width = 1920
height = 1080
fps = 15


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 1):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def load_mat(icam):
    load_file = experiment_root + 'L2-trajectories/L2_cam' + str(icam) + '.mat'
    trajectory = scipy.io.loadmat(load_file)
    data = trajectory['fileOutput']
    return data


def simple_data(all_data, icam):
    # store the np array from data
    # [:, 0]: id / [:, 1]: start time in camera1 / ... / [:, 8]: start time in carera8
    # because it will cost many time
    # so i save the nparray to id_data.npy
    '''    if os.path.isfile(file_name):
        id_data = np.load(file_name)
        return id_data
    else:
    '''
    file_name = visual_root + 'id_data' + str(icam) + '.npy'

    total_id = np.unique(all_data[:, 1])
    id_data = np.zeros((len(total_id), 2))
    for id_num in total_id:
        print(id_num)
        id_data[int(id_num) - 1, 0] = id_num
        data_new = [
            int(all_data[i, 0]) for i in range(len(all_data))
            if int(all_data[i, 1]) == id_num
        ]
        id_data[int(id_num) - 1, 1] = np.min(data_new)
    np.save(file_name, id_data)
    return id_data


def random_color(number_people):
    color = np.zeros((number_people + 1, 3))
    for i in range(0, number_people + 1):
        color[i] = list(np.random.choice(range(256), size=3))
    return color


def find_index(icam, frame, startFrame):
    window_size_bb = 80
    window_size_heatmap = 1000
    find_ind_heat = [
        i for i in range(len(data_part))
        if data_part[i][0] <= frame and data_part[i][0] >= frame -
        window_size_heatmap and data_part[i][0] >= startFrame
    ]
    find_ind_bb = [
        i for i in find_ind_heat if data_part[i][0] >= frame - window_size_bb
    ]
    find_ind = [i for i in find_ind_bb if data_part[i][0] == frame]
    return find_ind_bb, find_ind_heat, find_ind, len(find_ind)


def draw_bb(icam, frame, img, startFrame, find_ind):
    # draw the bounding box
    for i in find_ind:
        color_id = tuple(color[int(data_part[i][1])])
        left_x = int(data_part[i][2])
        left_y = int(data_part[i][3])
        right_x = int(data_part[i][2] + data_part[i][4])
        right_y = int(data_part[i][3] + data_part[i][5])
        if data_part[i][0] == frame:
            id_num = int(data_part[i][1])
            duaration_s = int(
                int(data_part[i][0] - id_data[id_num - 1, 1]) / fps)
            label_text = 'ID: ' + str(int(data_part[i][1]))
            duaration_text = str(duaration_s) + 's'
            if durations[id_num - 1, 2] < threshold_durations:
                duaration_text = duaration_text + ' (*)'
            cv2.rectangle(img, (left_x, left_y), (right_x, right_y), color_id,
                          2)
            cv2.rectangle(img, (left_x - 1, left_y - 20),
                          (right_x + 1, left_y), color_id, -1)
            cv2.putText(img, label_text, (left_x, left_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(img, duaration_text, (left_x, left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.circle(img, (int(data_part[i][2] + data_part[i][4] / 2), right_y),
                   2, color_id, -1)
    return img


def worldTomap(point_x, point_y):
    # get the map point
    return point_x, point_y


def draw_traj(icam, frame, find_ind):
    # draw the 2d location in the map
    img = cv2.imread(visual_root + 'map.jpg')
    for i in find_ind:
        label_text = str(int(data_part[i][1]))
        color_id = tuple(color[int(data_part[i][1])])
        px = int(data_part[i][2] + (data_part[i][4] / 2))
        py = int(data_part[i][3] + (data_part[i][5] / 2))
        cv2.circle(img, (px, py), 6, color_id, -1)
        cv2.putText(img, label_text, (px + 8, py + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_id, 1)
    return img


def cal_heatmap(icam, frame, startFrame, find_ind):
    # draw the image for heatmap
    heatmap_value = []
    path = visual_root + 'background' + str(icam) + '.jpg'
    background_img = Image.open(path)
    for i in find_ind:
        center_x = int(data_part[i][2] + (data_part[i][4] / 2))
        center_y = int(data_part[i][3] + data_part[i][5])
        heatmap_value.append((center_x, center_y))
    heatmapper = Heatmapper(point_diameter=30, point_strength=0.1, opacity=0.7)
    heatmap = heatmapper.heatmap_on_img(heatmap_value, background_img)
    img = cv2.cvtColor(np.asarray(heatmap), cv2.COLOR_RGB2BGR)
    return img


def cal_localtime(icam, frame_num):
    # get the real locat time
    start_sequence = 0
    return start_sequence + frame_num - start_time[icam - 1] + 1


def get_info_visitors(icam, start_time, global_time):
    data = load_mat(icam)
    file_name = visual_root + 'info_cam' + str(icam) + '.npy'
    if os.path.isfile(file_name):
        info_visitors = np.load(file_name)
    else:
        info_visitors = np.zeros((global_time - start_time, 2))
        for i in range(start_time, global_time):
            info_visitors[i - start_time, 0] = i
            info_visitors[i - start_time, 1] = len(
                [j for j in range(len(data)) if data[j, 0] == i])
        np.save(file_name, info_visitors)
    file_name2 = visual_root + 'info_duration' + str(icam) + '.npy'
    if os.path.isfile(file_name2):
        info_duration = np.load(file_name2)
    else:
        info_duration = np.zeros((len(set(data[:, 1])), 3))
        for i in range(1, len(set(data[:, 1])) + 1):
            time_min = min(
                [data[k, 0] for k in range(len(data)) if data[k, 1] == i])
            time_max = max(
                [data[k, 0] for k in range(len(data)) if data[k, 1] == i])
            info_duration[i - 1, 0] = time_min
            info_duration[i - 1, 1] = time_max
            info_duration[i - 1, 2] = time_max - time_min
        np.save(file_name2, info_duration)
    return info_duration


def cal_checkerboard(icam, data):
    file_name = visual_root + 'stay_cam' + str(icam) + '.npy'
    interval_x = np.zeros(board_A + 1)
    interval_y = np.zeros(board_B + 1)
    for i in range(len(interval_x)):
        interval_x[i] = i * (width / board_A)

    for i in range(len(interval_y)):
        interval_y[i] = i * (height / board_B)

    if os.path.isfile(file_name):
        stay_time = np.load(file_name)
    else:
        stay_time = np.zeros((len(set(data[:, 1])), board_A * board_B))
        for i in range(board_A):
            for j in range(board_B):
                for l in data:
                    px = int(l[2] + l[4] / 2)
                    py = int(l[3] + l[5])
                    id = int(l[1])
                    if interval_x[i] <= px < interval_x[i + 1] and interval_y[j] <= py < interval_y[j + 1]:
                        stay_time[id - 1, i * board_B + j] += 1
        np.save(file_name, stay_time)

    avg_board = np.mean(stay_time, axis=0)

    path = visual_root + 'background' + str(icam) + '.jpg'
    img = cv2.imread(path)

    for i in range(1, board_A):
        cv2.line(img, (int(interval_x[i]), 0), (int(interval_x[i]), 288), (0, 255, 255), 2)

    for i in range(1, board_B):
        cv2.line(img, (0, int(interval_y[i])), (360, int(interval_y[i])), (0, 255, 255), 2)

    for i in range(board_A):
        for j in range(board_B):
            text = '%.1f' % (avg_board[i * board_B + j] / fps)
            x_point = int(interval_x[i] + 5)
            y_point = int(interval_y[j + 1] - 5)
            cv2.putText(img, text, (x_point, y_point), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

    cv2.imshow('img', img)
    cv2.imwrite(visual_root + 'stay_img' + str(icam) + '.jpg', img)
    cv2.waitKey(1)


def main():
    startFrame_global = 0
    endFrame_global = 810

    for icam in range(1, cam_num + 1):
        # cam_num + 1):
        global durations
        durations = get_info_visitors(icam, startFrame_global, endFrame_global)
        data = load_mat(icam)
        # cal_checkerboard(icam, data)
        global data_part
        global color
        global id_data
        color = random_color(len(set(data[:, 1])))
        id_data = simple_data(data, icam)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_filename = experiment_root + 'video-results/camera' + str(
            icam) + '_result.avi'
        height = 1080
        width = 1920
        total_height = height
        # * 2 + 10 * 3
        total_width = width
        # * 2 + 10 * 3
        out = cv2.VideoWriter(out_filename, fourcc, fps,
                              (total_width, total_height))

        startFrame = cal_localtime(icam, startFrame_global)
        endFrame = cal_localtime(icam, endFrame_global)

        global data_part
        data_part = [
            data[i, :] for i in range(len(data))
            if data[i, 0] >= startFrame and data[i, 0] <= endFrame
        ]

        for current_frame in range(startFrame, endFrame):
            part_cam, part_frame = calucate_part(icam, current_frame)
            if current_frame == startFrame:
                filename = video_root + 'cam' + str(icam) + '.avi'
                cap = cv2.VideoCapture(filename)
            ret, frame_img = cap.read()

            find_ind_bb, find_ind_heat, find_ind, num_visitor = find_index(
                icam, current_frame, startFrame)

            # get the bounding box and put to image box 1
            frame_img = draw_bb(icam, current_frame, frame_img, startFrame,
                                find_ind_bb)

            cv2.putText(frame_img, str(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("video2", frame_img)
            cv2.waitKey(1)
            print('icam = ' + str(icam))
            print('frame = ' + str(current_frame) + ' / ' + str(endFrame))
            out.write(frame_img)
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
