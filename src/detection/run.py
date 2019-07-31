import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt
import scipy.io
import os

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--icam', type=int, default=1)
    parser.add_argument('--save_root', type=str)
    parser.add_argument('--video', type=str, default='./video/cam1.avi')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    detections = []
    icam = args.icam
    save_root = args.save_root
    cap = cv2.VideoCapture(args.video)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    for count in range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        print('frame = ', count)
        ret_val, image = cap.read()
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        for human in humans:
            temp = []
            temp.extend([icam, count])
            for i in range(0, 18):
                if i not in human.body_parts.keys():
                    temp.extend([0, 0, 0])
                else:
                    body_part = human.body_parts[i]
                    score = human.body_parts[i].score
                    # print('x = ', body_part.x, ' / y = ', body_part.y , ' / score = ', score)
                    temp.extend([body_part.x, body_part.y, score])
            print(temp)
            detections.append(temp)

        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (args.video, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # cv2.imshow('tf-pose-estimation result', image)
        # cv2.waitKey(1)

    detections = np.array(detections)
    detections = detections.reshape((len(detections), 56))  # 2d array of 3x3
    scipy.io.savemat(save_root + 'cam' + str(icam) + '_pose.mat', mdict={'detections': detections})
