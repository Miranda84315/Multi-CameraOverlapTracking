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
distCoeffs_c2 = distCoeffs_c1
np.save('datasets/resize/calibration/distCoeffs_c2.npy', distCoeffs_c2)

cameraMatrix_c3, distCoeffs_c3 = get_intrinsics('c3')
np.save('datasets/resize/calibration/cameraMatrix_c3.npy', cameraMatrix_c3)
distCoeffs_c3 = distCoeffs_c1
np.save('datasets/resize/calibration/distCoeffs_c3.npy', distCoeffs_c3)

cameraMatrix_c4, distCoeffs_c4 = get_intrinsics('c4')
np.save('datasets/resize/calibration/cameraMatrix_c4.npy', cameraMatrix_c4)
distCoeffs_c4 = distCoeffs_c1
np.save('datasets/resize/calibration/distCoeffs_c4.npy', distCoeffs_c4)

#   distortion


def undistortion():
    c_num = ['c1', 'c2', 'c3', 'c4']

    for i in range(0, 4):
        cameraMatrix = np.load('datasets\\resize\\calibration\\cameraMatrix_' + c_num[i] + '.npy')
        distCoeffs = np.load('datasets\\resize\\calibration\\distCoeffs_' + c_num[i] + '.npy')

        for j in range(1, 100):
            path = 'datasets\\resize\\data2\\' + c_num[i] + '\\' + str(j) + '.jpg'
            print(path)
            img = cv2.imread(path)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

            dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcameramtx)
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

#   -----------------------------------------------------------------------------------------------------------
#   for c1


def get_extrinsics_c1(c_num):
    objp = np.array([[-400, 40, 0], [-440, 40, 0], [-360, 80, 0], [-400, 80, 0], [-440, 80, 0],
                     [-320, 120, 0], [-360, 120, 0], [-400, 120, 0], [-440, 120, 0],
                     [0, 0, 0], [-40, 0, 0], [-80, 0, 0], [-120, 0, 0],
                     [0, 40, 0], [-40, 40, 0], [-80, 40, 0], [-120, 40, 0], [-160, 40, 0],
                     [-200, 40, 0], [-240, 40, 0], [-280, 40, 0], [-320, 40, 0], [-360, 40, 0],
                     [-40, 80, 0], [-80, 80, 0], [-120, 80, 0], [-160, 80, 0],
                     [-200, 80, 0], [-240, 80, 0], [-280, 80, 0], [-320, 80, 0],
                     [-80, 120, 0], [-120, 120, 0], [-160, 120, 0],
                     [-200, 120, 0], [-240, 120, 0], [-280, 120, 0]], np.float32)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    corners = np.array([[[224, 185]], [[211, 177]], [[208, 201]], [[196, 191]], [[183, 182]],
                        [[191, 221]], [[176, 208]], [[165, 197]], [[156, 187]],
                        [[562, 332]], [[486, 312]], [[448, 291]], [[413, 272]],
                        [[499, 356]], [[457, 332]], [[417, 308]], [[308, 287]], [[348, 266]],
                        [[319, 247]], [[295, 232]], [[270, 218]], [[253, 207]], [[237, 195]],
                        [[423, 356]], [[382, 328]], [[344, 303]], [[312, 280]],
                        [[285, 260]], [[261, 242]], [[240, 227]], [[221, 213]],
                        [[339, 349]], [[301, 320]], [[270, 294]], [[245, 271]], [[224, 253]], [[205, 236]]], np.float32)
    objpoints.append(objp)
    imgpoints.append(corners)

    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix_c1, distCoeffs_c1)

    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))

    objpoints_test = np.array([[-280, 120., 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t, cameraMatrix_c1, distCoeffs_c1)

    return Rt


#   -----------------------------------------------------------------------------------------------------------
#   for c2


def get_extrinsics_c2(c_num):
    objp = np.array([[0, 0, 0], [-40, 0, 0], [-80, 0, 0], [-120, 0, 0],
                     [160, 40, 0], [120, 40, 0], [80, 40, 0], [40, 40, 0], [0, 40, 0], [-40, 40, 0], [-80, 40, 0],
                     [-120, 40, 0], [-160, 40, 0], [-200, 40, 0], [-240, 40, 0], [-280, 40, 0], [-320, 40, 0],
                     [-360, 40, 0],
                     [160, 80, 0], [120, 80, 0], [80, 80, 0], [40, 80, 0], [0, 80, 0], [-40, 80, 0], [-80, 80, 0],
                     [-120, 80, 0], [-160, 80, 0], [-200, 80, 0], [-240, 80, 0], [-280, 80, 0], [-320, 80, 0],
                     [80, 120, 0], [40, 120, 0], [0, 120, 0], [-40, 120, 0], [-80, 120, 0],
                     [-120, 120, 0], [-160, 120, 0], [-200, 120, 0], [-240, 120, 0], [-280, 120, 0],
                     [80, 160, 0], [40, 160, 0], [0, 160, 0], [-40, 160, 0]], np.float32)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    corners = np.array([[[342, 282]], [[307, 285]], [[272, 288]], [[235, 291]],
                        [[480, 275]], [[453, 279]], [[424, 284]], [[390, 289]], [[353, 294]], [[318, 299]], [[278, 303]],
                        [[238, 305]], [[200, 308]], [[160, 307]], [[123, 308]], [[90, 309]], [[57, 307]], [[28, 302]],
                        [[497, 285]], [[470, 291]], [[440, 297]], [[406, 303]], [[368, 310]], [[328, 314]], [[286, 319]],
                        [[244, 323]], [[201, 325]], [[158, 325]], [[118, 325]], [[81, 323]], [[45, 323]],
                        [[459, 311]], [[424, 318]], [[384, 326]], [[341, 332]], [[296, 338]],
                        [[249, 343]], [[201, 345]], [[155, 346]], [[112, 346]], [[71, 345]],
                        [[481, 327]], [[443, 337]], [[402, 346]], [[356, 354]]], np.float32)
    objpoints.append(objp)
    imgpoints.append(corners)

    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix_c2, distCoeffs_c2)

    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))

    objpoints_test = np.array([[-280, 120., 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t, cameraMatrix_c2, distCoeffs_c2)

    return Rt


def get_extrinsics_c3(c_num):
    objp = np.array([[0, 0, 0], [-40, 0, 0], [-80, 0, 0], [-120, 0, 0],
                     [160, 40, 0], [120, 40, 0], [80, 40, 0], [40, 40, 0], [0, 40, 0], [-40, 40, 0], [-80, 40, 0],
                     [-120, 40, 0], [-160, 40, 0], [-200, 40, 0], [-240, 40, 0], [-280, 40, 0], [-320, 40, 0],
                     [-360, 40, 0],
                     [160, 80, 0], [120, 80, 0], [80, 80, 0], [40, 80, 0], [0, 80, 0], [-40, 80, 0], [-80, 80, 0],
                     [-120, 80, 0], [-160, 80, 0], [-200, 80, 0], [-240, 80, 0], [-280, 80, 0], [-320, 80, 0],
                     [80, 120, 0], [40, 120, 0], [0, 120, 0], [-40, 120, 0], [-80, 120, 0],
                     [-120, 120, 0], [-160, 120, 0], [-200, 120, 0], [-240, 120, 0], [-280, 120, 0],
                     [80, 160, 0], [40, 160, 0], [0, 160, 0], [-40, 160, 0]], np.float32)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    corners = np.array([[[511, 164]], [[490, 175]], [[468, 185]], [[444, 197]],
                        [[591, 141]], [[581, 147]], [[568, 155]], [[553, 164]], [[534, 175]], [[517, 186]], [[493, 199]],
                        [[468, 212]], [[436, 226]], [[403, 243]], [[364, 262]], [[320, 281]], [[275, 299]], [[225, 318]],
                        [[611, 148]], [[600, 156]], [[591, 164]], [[577, 175]], [[561, 186]], [[544, 197]], [[522, 211]],
                        [[497, 227]], [[467, 244]], [[432, 262]], [[391, 284]], [[347, 306]], [[297, 329]],
                        [[613, 174]], [[602, 186]], [[589, 198]], [[572, 210]], [[553, 225]],
                        [[530, 242]], [[500, 264]], [[466, 285]], [[426, 310]], [[378, 337]],
                        [[637, 185]], [[627, 196]], [[615, 210]], [[600, 225]]], np.float32)
    objpoints.append(objp)
    imgpoints.append(corners)

    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix_c3, distCoeffs_c3)

    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))

    objpoints_test = np.array([[9.4, 299.9, 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t, cameraMatrix_c3, distCoeffs_c3)

    return Rt


def get_extrinsics_c4(c_num):
    objp = np.array([[0, 0, 0], [-40, 0, 0], [-80, 0, 0], [-120, 0, 0],
                     [160, 40, 0], [120, 40, 0], [80, 40, 0], [40, 40, 0], [0, 40, 0], [-40, 40, 0], [-80, 40, 0],
                     [-120, 40, 0], [-160, 40, 0], [-200, 40, 0], [-240, 40, 0], [-280, 40, 0], [-320, 40, 0],
                     [-360, 40, 0],
                     [160, 80, 0], [120, 80, 0], [80, 80, 0], [40, 80, 0], [0, 80, 0], [-40, 80, 0], [-80, 80, 0],
                     [-120, 80, 0], [-160, 80, 0], [-200, 80, 0], [-240, 80, 0], [-280, 80, 0], [-320, 80, 0],
                     [80, 120, 0], [40, 120, 0], [0, 120, 0], [-40, 120, 0], [-80, 120, 0],
                     [-120, 120, 0], [-160, 120, 0], [-200, 120, 0], [-240, 120, 0], [-280, 120, 0],
                     [80, 160, 0], [40, 160, 0], [0, 160, 0], [-40, 160, 0]], np.float32)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    corners = np.array([[[291, 237]], [[264, 240]], [[237, 244]], [[209, 247]],
                        [[388, 230]], [[371, 233]], [[356, 237]], [[335, 242]], [[310, 246]], [[285, 251]], [[253, 255]],
                        [[222, 260]], [[190, 265]], [[154, 269]], [[121, 274]], [[84, 279]], [[50, 281]], [[16, 281]],
                        [[412, 239]], [[397, 243]], [[379, 247]], [[355, 254]], [[331, 258]], [[306, 264]], [[275, 270]],
                        [[242, 275]], [[207, 282]], [[169, 287]], [[130, 293]], [[92, 297]], [[52, 302]],
                        [[402, 258]], [[379, 265]], [[355, 271]], [[328, 277]], [[297, 284]],
                        [[264, 292]], [[226, 301]], [[187, 308]], [[114, 314]], [[102, 320]],
                        [[425, 270]], [[402, 277]], [[380, 284]], [[352, 292]]], np.float32)
    objpoints.append(objp)
    imgpoints.append(corners)

    objpoints2 = np.array(objpoints[0])
    imgpoints2 = np.array(imgpoints[0])
    retval, rvec, tvec = cv2.solvePnP(objpoints2, imgpoints2, cameraMatrix_c4, distCoeffs_c4)

    r = np.array(rvec)
    t = np.array(tvec)
    rvecs, jacobian = cv2.Rodrigues(r)
    Rt = np.hstack((rvecs, t))

    objpoints_test = np.array([[-280, 120., 0.]])
    r = Rt[:, 0:3]
    t = Rt[:, 3:]
    imagePoints, jacobian2 = cv2.projectPoints(objpoints_test, r, t, cameraMatrix_c4, distCoeffs_c4)

    return Rt
# for test
u = 182
v = 193
x, y = project_3d(u, v, cameraMatrix_c1, distCoeffs_c1, Rt_c1_new)

Rt_c1_new = get_extrinsics_c1('c1')
np.save('datasets/resize/calibration/Rt_c1_new.npy', Rt_c1_new)
Rt_c2 = get_extrinsics_c2('c2')
np.save('datasets/resize/calibration/Rt_c2.npy', Rt_c2)
Rt_c3 = get_extrinsics_c3('c3')
np.save('datasets/resize/calibration/Rt_c3.npy', Rt_c3)
Rt_c4 = get_extrinsics_c4('c4')
np.save('datasets/resize/calibration/Rt_c4.npy', Rt_c4)