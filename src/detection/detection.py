import numpy as np
import os
import cv2
import tensorflow as tf
import scipy.io
from object_detection.utils import label_map_util
'''
python detection.py --video_root D:/Code/AF_tracking/videos/ --save_root D:/Code/AF_tracking/dataset/detections/ --cam_num 4
'''

start_time = [1, 1, 1, 1]
start_sequence = 0
end_sequence = 225


def cal_localtime(icam, frame_num):
    # get the real locat time
    return frame_num - start_time[icam - 1] + 1


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
                    max_boxes_to_draw = 5
                    min_score_thresh = .50
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
                                feet_x = int(left + width/2)
                                feet_y = top + height
                                print(icam, frame, left, top, width, height,
                                      scores_new[i], feet_x, feet_y)
                                temp = [
                                    icam, frame_local + 1, left, top, width,
                                    height, scores_new[i], feet_x, feet_y
                                ]
                                detections.append(temp)
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


def main():
    cam_num = 1
    video_root = 'D:/Code/MultiCamOverlap/dataset/videos/No1/'
    save_root = 'D:/Code/MultiCamOverlap/dataset/detections/'
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

    object_detection(detection_graph, cam_num, video_root, save_root,
                     category_index)


if __name__ == '__main__':
    main()
