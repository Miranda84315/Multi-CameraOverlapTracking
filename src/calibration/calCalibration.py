import cv2
import numpy as np
import os
from scipy.linalg import solve
from scipy.optimize import fsolve
# -------------------------------------------------------------------   get c1~c4 Intrinsics


def get_intrinsics(c_num):
    w = 9
    h = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    file_dir = 'datasets/resize/calibration/' + c_num + '/'
    cameraMatrix = []
    distCoeffs = []
    count = 0
    for file_name in os.listdir(file_dir):
        print(file_dir + file_name)

        img = cv2.imread(file_dir + file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        # If found, add object points, image points (after refining them)

        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (h, w), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1)

    #   Calibration     find cameraMatrix and distCoeff
    retval, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                         None, None)

    # img = cv2.imread('datasets\\calibration\\temp\\c1\\14.jpg')
    # h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cmtx, dist, (w, h), 1, (w, h))
    #
    # dst = cv2.undistort(img, cmtx, dist, None, newcameramtx)
    # cv2.imshow('test', dst)
    # cv2.waitKey(1)
    cv2.destroyAllWindows()
    return cmtx, dist


cameraMatrix_c1, distCoeffs_c1 = get_intrinsics('c1')
np.save('datasets/resize/calibration/cameraMatrix_c1.npy', cameraMatrix_c1)
np.save('datasets/resize/calibration/distCoeffs_c1.npy', distCoeffs_c1)

cameraMatrix_c2, distCoeffs_c2 = get_intrinsics('c2')
np.save('datasets/resize/calibration/cameraMatrix_c2.npy', cameraMatrix_c2)
np.save('datasets/resize/calibration/distCoeffs_c2.npy', distCoeffs_c2)

cameraMatrix_c3, distCoeffs_c3 = get_intrinsics('c3')
np.save('datasets/resize/calibration/cameraMatrix_c3.npy', cameraMatrix_c3)
np.save('datasets/resize/calibration/distCoeffs_c3.npy', distCoeffs_c3)

cameraMatrix_c4, distCoeffs_c4 = get_intrinsics('c4')
np.save('datasets/resize/calibration/cameraMatrix_c4.npy', cameraMatrix_c4)
np.save('datasets/resize/calibration/distCoeffs_c4.npy', distCoeffs_c4)

def load_cameramtx():
    cameraMatrix_c1 = np.load('datasets\\calibration\\cameraMatrix_c1.npy')
    distCoeffs_c1 = np.load('datasets\\calibration\\distCoeffs_c1.npy')
    Rt_c1 = np.load('datasets\\calibration\\Rt_c1.npy')

    cameraMatrix_c2 = np.load('datasets\\calibration\\cameraMatrix_c2.npy')
    distCoeffs_c2 = np.load('datasets\\calibration\\distCoeffs_c2.npy')
    Rt_c2 = np.load('datasets\\calibration\\Rt_c2.npy')

    cameraMatrix_c3 = np.load('datasets\\calibration\\cameraMatrix_c3.npy')
    distCoeffs_c3 = np.load('datasets\\calibration\\distCoeffs_c3.npy')
    Rt_c3 = np.load('datasets\\calibration\\Rt_c3.npy')

    cameraMatrix_c4 = np.load('datasets\\calibration\\cameraMatrix_c4.npy')
    distCoeffs_c4 = np.load('datasets\\calibration\\distCoeffs_c4.npy')
    Rt_c4 = np.load('datasets\\calibration\\Rt_c4.npy')

# -------------------------------------------------------------------   get c1~c4 Extrinsics

def get_extrinsics(c_num):
    w = 9
    h = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    file_dir = 'datasets/calibration_world/test/' + c_num + '.jpg'
    cameraMatrix = []
    distCoeffs = []
    count = 0

    img = cv2.imread(file_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
    # If found, add object points, image points (after refining them)

    if ret is True:
        count += 1
        print('count = '. count)

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (h, w), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1)
    else:
        print('false')


    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix_c2, distCoeffs_c2)

    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))

    objpoints_test = np.array([[5, 0., 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t, cameraMatrix_c2, distCoeffs_c2)

    return Rt



Rt_c1 = get_extrinsics('c1')
np.save('datasets/calibration/new_calibration/Rt_c1.npy', Rt_c1)
Rt_c2 = get_extrinsics('c2')
np.save('datasets/calibration/new_calibration/Rt_c2.npy', Rt_c2)
Rt_c3 = get_extrinsics('c3')
np.save('datasets/calibration/new_calibration/Rt_c3.npy', Rt_c3)
Rt_c4 = get_extrinsics('c4')
np.save('datasets/calibration/new_calibration/Rt_c4.npy', Rt_c4)
#   distortion

def undistortion():
    c_num = ['c1', 'c2', 'c3', 'c4']

    for i in range(0, 4):
        cameraMatrix = np.load('datasets\\calibration\\cameraMatrix_' + 'c1' + '.npy')
        distCoeffs = np.load('datasets\\calibration\\distCoeffs_' + 'c1' + '.npy')

        for j in range(1, 25):
            path = 'datasets\\all_data\\' + c_num[i] + '\\' + str(j) + '.jpg'
            print(path)
            img = cv2.imread(path)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

            dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
            cv2.imwrite('datasets\\all_data\\undistortion\\' + c_num[i] + '\\' + str(j) + '.jpg', dst)
            cv2.imshow('test', dst)
            cv2.waitKey(1)

    cv2.destroyAllWindows()


#   by ugly
def project_3d(Rt, cameraMatrix, distCoeffs):
    obj_3dPoint = np.array([[5], [0], [0], [1]])

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]

    k1 = distCoeffs[0][0]
    k2 = distCoeffs[0][1]
    k3 = distCoeffs[0][4]
    p1 = distCoeffs[0][2]
    p2 = distCoeffs[0][3]

    temp_Rt = np.dot(Rt, obj_3dPoint)

    x = temp_Rt[0][0]
    y = temp_Rt[1][0]
    z = temp_Rt[2][0]

    x_one = x / z
    y_one = y / z

    r2 = x_one * x_one + y_one * y_one
    r4 = np.power(x_one, 4) + np.power(y_one, 4)
    r6 = np.power(x_one, 6) + np.power(y_one, 6)

    ugly = 1 + k1 * r2 + k2 * r4 + k3 * r6

    x_two = x_one * ugly + 2 * p1 * x_one * y_one + p2 * (r2 + 2 * x_one * x_one)
    u = fx*x_two + cx

    y_two = y_one * ugly + p1 * (r2 + 2 * y_one * y_one) + 2 * p2 * x_one * y_one
    v = fy*y_two + cy

    return u, v

#   ------------------------------------------------ get c2 (u,v,1) = (X, Y, 0)


# a, b = project_3d(407.02393584, 384.161578, cameraMatrix_c1, distCoeffs_c1, Rt)


def project_3d(u, v, cameraMatrix, distCoeffs, Rt):
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]

    k1 = distCoeffs[0][0]
    k2 = distCoeffs[0][1]
    k3 = distCoeffs[0][4]
    p1 = distCoeffs[0][2]
    p2 = distCoeffs[0][3]

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

