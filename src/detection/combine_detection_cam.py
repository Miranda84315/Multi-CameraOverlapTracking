import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import scipy.cluster.hierarchy as hcluster
import cv2
import os
from copkmeans.cop_kmeans import cop_kmeans
from argparse import ArgumentParser
'''
use Spectral Clustering to combined detections
combine_detection.py --track 2/ --endFrame 390
'''

parser = ArgumentParser(description='combine detection')
parser.add_argument('--track', type=str, required=True, help='Input cam number')
parser.add_argument('--calibration', type=str, required=True, help='Input cam number')
parser.add_argument('--save', type=str, required=True, help='Input cam number')
# parser.add_argument('--endFrame', type=int, required=True, help='Input cam number')

args = parser.parse_args()

detection_dir = args.save
track = args.track
# endFrame = args.endFrame
matrix_save = args.calibration

'''
matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0315/information/'
detection_dir = 'D:/Code/MultiCamOverlap/dataset/alpha_pose/Player01/track'
track = '1/'
'''
version = 0
'''
    version 0: experiments_alpla - Outlier + spectral & constraint + constrained K-Means (Best)
    version 1: experiments_cluster1 - Spectral
    version 2: experiments_cluster2 - Outlier + spectral
    version 3: experiments_cluster3 - Outlier + spectral & constraint
    version 4: experiments_cluster4 - Outlier + constrained K-Means
'''

detection_root = detection_dir + track
save_root = detection_dir + track
save_img = detection_root + 'imgcam12/'

cam_num = 4
cam_total = 2
width = 1920
height = 1080
color = ['bo', 'go', 'ro', 'co', 'mo', 'ko', 'yo', 'bo', 'go', 'ro', 'co', 'mo', 'ko']


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_detection(cam_num):
    detections = []
    num_each_camera = [0]
    min_frame = np.inf
    temp = 0
    for icam in range(1, cam_num + 1):
        load_file = detection_root + 'cam' + str(icam) + '.mat'
        data = scipy.io.loadmat(load_file)
        data = data['detections']
        num_each_camera.append(temp + len(data))
        temp = temp + len(data)
        temp_frame = max(data[:, 1])
        if temp_frame < min_frame:
            min_frame = temp_frame
        detections.extend(data)
    return np.array(detections), num_each_camera, int(min_frame)


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
    return distance
    # 1 / (1 + distance)


def getSimilarityMatrix(x):
    num_x = len(x)
    similarityMatrix = np.zeros((num_x, num_x))
    for i in range(0, num_x):
        for j in range(i, num_x):
            x1 = x[i, 1]
            x2 = x[j, 1]
            y1 = x[i, 2]
            y2 = x[j, 2]
            similarityMatrix[i, j] = 1/(1+calucate_distance(x1, y1, x2, y2)/200)
            if (x[i, 0] == x[j, 0]) and (i != j):
                similarityMatrix[i, j] = 0
            if i == j:
                similarityMatrix[i, j] = 1
            similarityMatrix[j, i] = similarityMatrix[i, j]
    return similarityMatrix


def getSimilarityMatrix_noconstraint(x):
    num_x = len(x)
    similarityMatrix = np.zeros((num_x, num_x))
    for i in range(0, num_x):
        for j in range(i, num_x):
            x1 = x[i, 1]
            x2 = x[j, 1]
            y1 = x[i, 2]
            y2 = x[j, 2]
            similarityMatrix[i, j] = 1/(1+calucate_distance(x1, y1, x2, y2)/200)
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


def nearestPoint(x, y):
    nearest = []
    for i in range(0, len(x)):
        min_dis = np.inf
        for j in range(0, len(x)):
            if i != j:
                if calucate_distance(x[i], y[i], x[j], y[j]) < min_dis:
                    min_dis = calucate_distance(x[i], y[i], x[j], y[j])
        nearest.append(int(min_dis))
    nearest = np.array(nearest)
    return nearest


def main():
    startFrame = 0
    plot_img = False

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

    detections, num_each_camera, endFrame = load_detection(cam_num)
    # endFrame = int(max(detections[:, 1]))
    total_detections = []
    for current_frame in range(startFrame, endFrame):
        detection = np.array([detections[i, [0, 7, 8]] for i in range(len(detections)) if (detections[i, 1] == (current_frame + 1)) and (int(detections[i, 0]) == 3 or int(detections[i, 0]) == 4) ])
        for i in range(0, len(detection)):
            # plot feet_x, feet_y into 3D location
            x, y = project_3d(detection[i, 1], detection[i, 2], cmtx, dist, Rt[int(detection[i, 0]) - 1])
            detection[i, 1] = x
            detection[i, 2] = y

        original_index = np.array([i for i in range(len(detections)) if (detections[i, 1] == (current_frame + 1)) and (int(detections[i, 0]) == 3 or int(detections[i, 0]) == 4) ])
        # delete lonely point
        nearest = nearestPoint(detection[:, 1], detection[:, 2])
        detection = detection[nearest <= 150, :]
        original_index = original_index[nearest <= 150]

        # 1. get similarity matrix  2. count max camera's detection 3. spectral clustering and get labels 
        # if plot_img is True:
        #    plt.scatter(*np.transpose(detection[:, 1:3]), c=clusters)
        #    plt.axis("equal")
        #    title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        #    plt.title(title)
        #    plt.show()

        # -- (1) use spectral clustering
        # similarity_Matrix = getSimilarityMatrix(detection)
        # sc = SpectralClustering(n_clusters=5, affinity='precomputed', n_init=10)
        # sc.fit(similarity_Matrix)
        # -- (2) use Hierarchical Clustering, directly using detection's xy point
        # sc = AgglomerativeClustering(n_clusters=true_counts)
        # sc.fit(detection[:, 1:3])
        # label = np.array(sc.labels_).reshape((len(detection), 1))
        # count_num, counts = np.unique(label, return_counts=True)
        # -- (3) still use fclusterdata, but divide the error clustering.

        # -------------- version 1
        if version == 0:
            if len(detection) >= 1:
                _, counts = np.unique(detection[:, 0], return_counts=True)
                thresh = 200
                clusters = hcluster.fclusterdata(detection[:, 1:3], thresh, criterion="distance")

                _, counts2 = np.unique(clusters, return_counts=True)
                current_cluster_num = len(counts2)

                for ind in counts2:
                    if ind > cam_total:
                        current_cluster_num += int(np.ceil(ind/cam_total)) - 1

                similarity_Matrix = getSimilarityMatrix(detection)
                sc = SpectralClustering(n_clusters=current_cluster_num, affinity='precomputed', n_init=20, assign_labels="discretize")
                sc.fit(similarity_Matrix)
                _, counts2 = np.unique(sc.labels_, return_counts=True)
                label = np.array(sc.labels_).reshape((len(detection), 1))

                if any(counts2 >= cam_total + 1):
                    print(current_frame)
                    must_x, must_y = np.where(similarity_Matrix == 1)
                    not_x, not_y = np.where(similarity_Matrix == 0)
                    must_zipped = list(zip(must_x, must_y))
                    not_zipped = list(zip(not_x, not_y))
                    clusters, centers = cop_kmeans(dataset=similarity_Matrix, k=current_cluster_num, ml=must_zipped, cl=not_zipped)
                    if clusters is not None:
                        label = np.array(clusters).reshape((len(detection), 1))

        # -------------- version 1 end

        # combined detection and label
        if len(detection) >= 1:
            detection = np.append(detection, label, axis=1)
            # calucate new detection using mean x, y-point.
            new_detection = []
            # print(label_num)
            for i in np.unique(label):
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
                # for i in range(0, len(new_detection)):
                #    plt.plot(new_detection[i, 1], new_detection[i, 2], color[int(detection[i, 3])])
                # -- << plot new detection with triangle point >>
                plt.plot(new_detection[:, 1], new_detection[:, 2], 'y^')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim((0, 1500))
                plt.ylim((1400, 0))
                # plt.legend(['bo', 'go', 'ro', 'co'], ['cam1', 'cam2', 'cam3', 'cam4'], loc='upper left')
                plt.title('clustering')
                plt.savefig(save_img + str(current_frame+1) + '.png')
                # plt.show()
                plt.close()

    total_detections = np.array(total_detections).reshape((len(total_detections), 7))
    # print(total_detections)
    # scipy.io.savemat(save_root + 'camera_all.mat', mdict={'detections': total_detections})
    scipy.io.savemat(save_root + 'camera_cam34.mat', mdict={'detections': total_detections})
    # scipy.io.savemat(save_root + 'camera_cam124.mat', mdict={'detections': total_detections})
    # scipy.io.savemat(save_root + 'camera_cam134.mat', mdict={'detections': total_detections})


if __name__ == '__main__':
    main()
