import numpy as np
import os
import cv2
import tensorflow as tf
import scipy.io
from object_detection.utils import label_map_util
from argparse import ArgumentParser

'''
python detection.py --video_root D:/Code/AF_tracking/videos/ --save_root D:/Code/AF_tracking/dataset/detections/ --cam_num 4
'''

parser = ArgumentParser(description='Objection Detection.')

parser.add_argument(
    '--video_root', required=True, help='Input video location.')

parser.add_argument(
    '--save_root', required=True, help='Save the result mat location.')

parser.add_argument(
    '--cam_num', required=True, default=8, help='camera number.')

parser.add_argument(
    '--model_name',
    default='faster_rcnn_inception_v2_coco_2018_01_28',
    help='select model to do detection')

parser.add_argument(
    '--start_time_regular',
    default=False,
    help='If your video frame 1 is same time, then true or false')

start_time = [1, 1, 1, 1]
NumFrames = [5010, 5010, 5010, 5010]
PartFrames = [[5010, 5010, 5010, 5010]]
start_sequence = 0
end_sequence = 5010


def calucate_part(icam, frame):
    sum_frame = 0
    for part_num in range(0, 1):
        previs_sum = sum_frame
        sum_frame += PartFrames[part_num][icam - 1]
        if sum_frame >= frame + 1:
            return part_num, frame - previs_sum


def cal_localtime(icam, frame_num):
    # get the real locat time
    return frame_num - start_time[icam - 1] + 1


def object_detection(detection_graph, args, category_index):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            for icam in range(1, int(args.cam_num)+1):
                part_cam_previous = -1
                detections = []
                for frame in range(start_sequence, end_sequence):
                    frame_local = cal_localtime(icam, frame)
                    part_cam, part_frame = calucate_part(icam, frame_local)

                    if frame == start_sequence:
                        filename = args.video_root + 'camera' + str(
                            icam) + '/0000' + str(part_cam) + '.avi'
                        cap = cv2.VideoCapture(filename)
                        cap.set(1, part_frame)
                        part_cam_previous = part_cam
                    if part_cam != part_cam_previous:
                        filename = args.video_root + 'camera' + str(
                            icam) + '/0000' + str(part_cam) + '.avi'
                        cap = cv2.VideoCapture(filename)
                    part_cam_previous = part_cam

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
                    min_score_thresh = .10
                    for i in range(min(max_boxes_to_draw, boxes_new.shape[0])):
                        if scores_new is None or (scores_new[i] > min_score_thresh):
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
                                (left, right, top, bottom) = (
                                            int(test_box[1] * width),
                                            int(test_box[3] * width),
                                            int(test_box[0] * height),
                                            int(test_box[2] * height))
                                (left, top, width, height) = (left, top, right - left, bottom - top)
                                frame_img = cv2.rectangle(frame_img, (left, top), (right, bottom), (0, 255, 0), 2)

                                print(icam, frame, left, top, width, height,scores_new[i])
                                temp = [icam, frame_local+1, left, top, width, height, scores_new[i]]
                                detections.append(temp)
                    cv2.imshow("video", frame_img)
                    cv2.waitKey(1)
                    print(frame)
                detections = np.array(detections)
                detections = detections.reshape((len(detections), 7))  # 2d array of 3x3
                scipy.io.savemat(args.save_root + 'camera' + str(icam) + '.mat', mdict={'detections': detections})

    cv2.destroyAllWindows()


def main():
    args = parser.parse_args()

    # MODEL_NAME
    MODEL_NAME = args.model_name
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

    object_detection(detection_graph, args, category_index)


if __name__ == '__main__':
    main()
