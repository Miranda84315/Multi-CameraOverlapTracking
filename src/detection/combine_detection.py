import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import scipy.cluster.hierarchy as hcluster
import cv2
import os
from argparse import ArgumentParser
'''
use Spectral Clustering to combined detections
combine_detection.py --track 2/ --endFrame 390
'''

parser = ArgumentParser(description='combine detection')
parser.add_argument('--track', type=str, required=True, help='Input cam number')
parser.add_argument('--calibration', type=str, required=True, help='Input cam number')
parser.add_argument('--save', type=str, required=True, help='Input cam number')
parser.add_argument('--endFrame', type=int, required=True, help='Input cam number')

args = parser.parse_args()


detection_dir = args.save
track = args.track
# matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0315/information/'
# detection_dir = 'D:/Code/MultiCamOverlap/dataset/detections/Player01/track'
# '2/'

detection_root = detection_dir + track
matrix_save = args.calibration
save_root = detection_dir + track
save_img = detection_root + 'img/'

cam_num = 4
width = 1920
height = 1080
color = ['bo', 'go', 'ro', 'co', 'mo', 'ko']


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_detection(cam_num):
    detections = []
    num_each_camera = [0]
    temp = 0
    for icam in range(1, cam_num + 1):
        load_file = detection_root + 'cam' + str(icam) + '.mat'
        data = scipy.io.loadmat(load_file)
        data = data['detections']
        num_each_camera.append(temp + len(data))
        temp = temp + len(data)
        detections.extend(data)
    return np.array(detections), num_each_camera


#   by ugly
# 2D to 3D
def project_3d(u, v, cameraMatrix, distCoeffs, Rt):
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]
    k1 = distCoeffs[0]
    k2 = distCoeffs[1]
    k3 = distCoeffs[4]
    p1 = distCoeffs[2]
    p2 = distCoeffs[3]
    x_two = (u - cx) / fx
    y_two = (v - cy) / fy

    def f1(x):
        x_one = float(x[0])
        y_one = float(x[1])
        r2 = x_one * x_one + y_one * y_one
        r4 = x_one * x_one * x_one * x_one + y_one * y_one * y_one * y_one
        r6 = x_one * x_one * x_one * x_one * x_one * x_one + y_one * y_one * y_one * y_one * y_one * y_one
        return [x_two - (x_one * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x_one * y_one + p2 * (r2 + 2 * x_one * x_one)),
                y_two - (y_one * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2 * y_one * y_one) + 2 * p2 * x_one * y_one)]
    [x_one, y_one] = fsolve(f1, [0, 0])

    def f2(x):
        X = float(x[0])
        Y = float(x[1])
        return [x_one - ((Rt[0][0] * X + Rt[0][1] * Y + Rt[0][3]) / (Rt[2][0] * X + Rt[2][1] * Y + Rt[2][3])),
                y_one - ((Rt[1][0] * X + Rt[1][1] * Y + Rt[1][3]) / (Rt[2][0] * X + Rt[2][1] * Y + Rt[2][3]))]
    [world_x, world_y] = fsolve(f2, [0, 0])
    return world_x, world_y


# 3D to 2D
def point3Dto2D(world_x, world_y, cameraMatrix, distCoeffs, Rt):
    world_point = np.array([[world_x, world_y, 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(world_point, r, t, cameraMatrix, distCoeffs)
    img_x = imagePoints[0][0][0]
    img_y = imagePoints[0][0][1]
    return img_x, img_y


def calucate_distance(x1, y1, x2, y2):
    distance = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return 1 / (1 + distance)


def getSimilarityMatrix(x):
    num_x = len(x)
    similarityMatrix = np.zeros((num_x, num_x))
    for i in range(0, num_x):
        for j in range(i, num_x):
            x1 = x[i, 1]
            x2 = x[j, 1]
            y1 = x[i, 2]
            y2 = x[j, 2]
            similarityMatrix[i, j] = calucate_distance(x1, y1, x2, y2)
            if (x[i, 0] == x[j, 0]) and (i != j):
                similarityMatrix[i, j] = 0
            if i == j:
                similarityMatrix[i, j] = 1
            similarityMatrix[j, i] = similarityMatrix[i, j]

    return similarityMatrix


def getMeanPoint(point):
    return np.mean(point, axis=0)


def recomputeIndex(index, n, each_n):
    new_index = np.zeros(n, dtype=int)
    new_index[:] = -1
    for i in index:
        for j in range(1, n + 1):
            if i < each_n[j]:
                new_index[j-1] = i - each_n[j - 1] + 1
                break
    return new_index


def main():
    startFrame = 0
    endFrame = args.endFrame
    plot_img = True

    cmtx = np.loadtxt(matrix_save + 'intrinsics.txt')
    dist = np.loadtxt(matrix_save + 'distCoeffs.txt')
    Rt = []

    createFolder(save_root)
    createFolder(save_img)

    for i in range(1, cam_num + 1):
        Rt_temp = np.loadtxt(matrix_save + 'Rt' + str(i) + '.txt')
        Rt.append(Rt_temp)

    # detections: total_detection
    # detection: detections[frame == current_frame ]
    # info_detection: [frame, x, y, cam1, cam2, cam3, cam4] x, y is mean of the same label clustering
    # cam1~cam4: the x y is from original detecions_cam1[cam1], ... , detections_cam4[cam4]
    # cam1~cam4: if is -1, then it is no detection from this camera
    # new_detection: for plt, record this frame's clustering result

    detections, num_each_camera = load_detection(cam_num)
    total_detections = []
    for current_frame in range(startFrame, endFrame):
        print(current_frame)
        detection = np.array([detections[i, [0, 7, 8]] for i in range(len(detections)) if detections[i, 1] == (current_frame + 1)])
        original_index = np.array([i for i in range(len(detections)) if detections[i, 1] == (current_frame + 1)])
        for i in range(0, len(detection)):
            # plot feet_x, feet_y into 3D location
            x, y = project_3d(detection[i, 1], detection[i, 2], cmtx, dist, Rt[int(detection[i, 0]) - 1])
            detection[i, 1] = x
            detection[i, 2] = y

        # 1. get similarity matrix  2. count max camera's detection 3. spectral clustering and get labels 
        _, counts = np.unique(detection[:, 0], return_counts=True)

        thresh = 160
        clusters = hcluster.fclusterdata(detection[:, 1:3], thresh, criterion="distance")
        #if plot_img is True:
        #    plt.scatter(*np.transpose(detection[:, 1:3]), c=clusters)
        #    plt.axis("equal")
        #    title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        #    plt.title(title)
        #    plt.show()

        fcluster_count = len(set(clusters))
        _, counts2 = np.unique(clusters, return_counts=True)
        for i in counts2:
            if i > 4:
                fcluster_count += 1

        true_counts = 5 if max(counts) == 5 else fcluster_count
        if true_counts > 5:
            true_counts = 5
        # -- (1) use spectral clustering
        # similarity_Matrix = getSimilarityMatrix(detection)
        # sc = SpectralClustering(n_clusters=5, affinity='precomputed', n_init=10)
        # sc.fit(similarity_Matrix)

        # -- (2) use Hierarchical Clustering, directly using detection's xy point
        sc = AgglomerativeClustering(n_clusters=true_counts)
        sc.fit(detection[:, 1:3])
        label = np.array(sc.labels_).reshape((len(detection), 1))
        _, counts = np.unique(label, return_counts=True)
        #print(counts)
        if max(counts) >= 5:
            true_counts += 1
            sc = AgglomerativeClustering(n_clusters=true_counts)
            sc.fit(detection[:, 1:3])
            label = np.array(sc.labels_).reshape((len(detection), 1))
        # combined detection and label
        detection = np.append(detection, label, axis=1)

        # calucate new detection using mean x, y-point.
        new_detection = []
        for i in range(0, true_counts):
            info_detection = [current_frame + 1]
            coordinate = np.array([detection[k, 1:3] for k in range(0, len(detection)) if detection[k, 3] == i])
            index = [original_index[k] for k in range(0, len(original_index)) if label[k] == i]
            index = recomputeIndex(index, cam_num, num_each_camera)
            info_detection.extend(getMeanPoint(coordinate))
            info_detection.extend(index)
            new_detection.append(info_detection)
            total_detections.append(info_detection)

        # plot result after clustering
        if plot_img is True:
            new_detection = np.array(new_detection)
            # -- << plot each original detection >>
            for i in range(0, len(detection)):
                plt.plot(detection[i, 1], detection[i, 2], color[int(detection[i, 3])])
            # -- << plot new detection with circle point >>
            #for i in range(0, len(new_detection)):
            #    plt.plot(new_detection[i, 1], new_detection[i, 2], color[int(detection[i, 3])])
            # -- << plot new detection with triangle point >>
            plt.plot(new_detection[:, 1], new_detection[:, 2], 'y^')
            plt.xlabel('x')
            plt.ylabel('y')
            #plt.legend(['bo', 'go', 'ro', 'co'], ['cam1', 'cam2', 'cam3', 'cam4'], loc='upper left')
            plt.title('clustering')
            plt.savefig(save_img + str(current_frame) + '.png')
            #plt.show()
            plt.close()

    total_detections = np.array(total_detections).reshape((len(total_detections), 7))
    print(total_detections)
    scipy.io.savemat(save_root + 'camera_all.mat', mdict={'detections': total_detections})


if __name__ == '__main__':
    main()
