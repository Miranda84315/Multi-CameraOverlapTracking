import numpy as np
import os
import cv2
import tensorflow as tf
import scipy.io
from object_detection.utils import label_map_util
from matplotlib.path import Path
'''
python detection.py --video_root D:/Code/AF_tracking/videos/ --save_root D:/Code/AF_tracking/dataset/detections/new_delete_other/ --cam_num 4
'''

start_time = [1, 1, 1, 1]
start_sequence = 0
end_sequence = 0

calibration_dir = 'D:/Code/MultiCamOverlap/dataset/calibration/0317/information/'
#video_dir = 'D:/Code/MultiCamOverlap/dataset/videos/Player01/track'
#save_dir = 'D:/Code/MultiCamOverlap/dataset/detections/Player01/track'
#track_num = '4/'
track_num = ['1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/']
'''
p1 = Path([(321, 685), (1605, 644), (1918, 731), (1914, 1075), (0, 1080), (0, 794)])
p2 = Path([(2, 558), (795, 520), (1858, 677), (1691, 1077), (0, 1075)])
p3 = Path([(0, 455), (792, 388), (1905, 738), (1391, 1077), (0, 1072)])
p4 = Path([(51, 478), (462, 1074), (811, 1075), (1732, 658), (921, 484)])
p = [p1, p2, p3, p4]'''
roi_filename = calibration_dir + 'ROI.npy'
p = np.load(roi_filename)


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_frame_number(cam_num, video_root):
    frame = []
    for icam in range(1, cam_num + 1):
        filename = video_root + 'cam' + str(icam) + '.avi'
        cap = cv2.VideoCapture(filename)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame.append(frame_num)
    return min(frame)


def cal_localtime(icam, frame_num):
    # get the real locat time
    return frame_num - start_time[icam - 1] + 1


def inROI(icam, x, y):
    return p[icam - 1].contains_points([(x, y)])[0]


def object_detection(detection_graph, cam_num, video_root, save_root,
                     category_index):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            for icam in range(1, cam_num + 1):
                detections = []
                for frame in range(start_sequence, end_sequence):
                    frame_local = cal_localtime(icam, frame)
                    if frame == start_sequence:
                        filename = video_root + 'cam' + str(icam) + '.avi'
                        cap = cv2.VideoCapture(filename)

                    # load video frame
                    ret, frame_img = cap.read()

                    image_np_expanded = np.expand_dims(frame_img, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name(
                        'image_tensor:0')
                    # Each box represents a part of the image
                    boxes = detection_graph.get_tensor_by_name(
                        'detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name(
                        'detection_scores:0')
                    classes = detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    boxes_new = np.squeeze(boxes)
                    classes_new = np.squeeze(classes).astype(np.int32)
                    scores_new = np.squeeze(scores)
                    category_index_new = category_index
                    max_boxes_to_draw = 10
                    min_score_thresh = .85
                    for i in range(min(max_boxes_to_draw, boxes_new.shape[0])):
                        if scores_new is None or (scores_new[i] >
                                                  min_score_thresh):
                            test1 = None
                            test2 = None

                            if category_index_new[classes_new[i]]['name']:
                                test1 = category_index_new[classes_new[i]][
                                    'name']
                                test2 = int(100 * scores_new[i])

                            line = {}
                            line[test1] = test2
                            #   we only do which detection class is person
                            if test1 == "person":
                                test_box = boxes_new[i]

                                #   get detection's left, right, top, bottom
                                height, width = frame_img.shape[:2]
                                (left, right, top,
                                 bottom) = (int(test_box[1] * width),
                                            int(test_box[3] * width),
                                            int(test_box[0] * height),
                                            int(test_box[2] * height))
                                (left, top, width,
                                 height) = (left, top, right - left,
                                            bottom - top)
                                frame_img = cv2.rectangle(
                                    frame_img, (left, top), (right, bottom),
                                    (0, 255, 0), 2)
                                feet_x = int(left + width / 2)
                                feet_y = top + height
                                # print(icam, frame, left, top, width, height, scores_new[i], feet_x, feet_y)
                                temp = [
                                    icam, frame_local + 1, left, top, width,
                                    height, scores_new[i], feet_x, feet_y
                                ]
                                roi = inROI(icam, feet_x, feet_y)
                                if roi:
                                    detections.append(temp)
                                else:
                                    print('out')
                    cv2.imshow("video", frame_img)
                    cv2.waitKey(1)

                    print(frame)
                detections = np.array(detections)
                detections = detections.reshape((len(detections),
                                                 9))  # 2d array of 3x3
                scipy.io.savemat(
                    save_root + 'cam' + str(icam) + '.mat',
                    mdict={'detections': detections})
    cv2.destroyAllWindows()


def main(player, track):
    global video_dir, save_dir
    cam_num = 4
    video_root = video_dir[player] + track_num[track]
    save_root = save_dir[player] + track_num[track]
    createFolder(video_root)
    createFolder(save_root)
    # MODEL_NAME
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    # Path to frozen detection graph. This is the actual model
    # - that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data_object', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    global end_sequence
    end_sequence = get_frame_number(cam_num, video_root)
    object_detection(detection_graph, cam_num, video_root, save_root,
                     category_index)


if __name__ == '__main__':
    global video_dir
    global save_dir
    video_dir = [
        'D:/Code/MultiCamOverlap/dataset/videos/Player08/track',
        'D:/Code/MultiCamOverlap/dataset/videos/Player20/track',
        'D:/Code/MultiCamOverlap/dataset/videos/Player21/track'
    ]
    save_dir = [
        'D:/Code/MultiCamOverlap/dataset/detections/Player08/track',
        'D:/Code/MultiCamOverlap/dataset/detections/Player20/track',
        'D:/Code/MultiCamOverlap/dataset/detections/Player21/track'
    ]
    for player in range(0, 1):
        print('player: ', player)
        for track in range(0, 8):
            print('track : ', track)
            # main(player, track)
            system_cmd = 'C:/Users/Owner/Anaconda3/envs/tensorflow/python.exe combine_detection.py --track ' + track_num[track] + ' --calibration ' + calibration_dir + ' --save ' + save_dir[player]
            print(system_cmd)
            os.system(system_cmd)
