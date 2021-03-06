import numpy as np

'''
np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0],
     [0, 1400, 0], [1500, 1400, 0]], np.float32)

np.array(
    [[[, ]], [[, ]], [[, ]], [[, ]], [[, ]], [[, ]],
     [[, ]], [[, ]], [[, ]], [[, ]],
     [[, ]], [[, ]], [[, ]], [[, ]], 
     [[, ]], [[, ]]], np.float32)
'''

matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0421_36/information/'
cam_num = 4


objp_cam1 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam1 = np.array(
    [[[350, 591]], [[417, 591]], [[764, 584]], [[1208, 572]], [[1551, 562]], [[1612, 557]], 
[[732, 613]], [[682, 657]], [[655, 681]], [[627, 704]], 
[[1243, 598]], [[1297, 637]], [[1324, 657]], [[1358, 684]]], np.float32)

objp_cam2 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam2 = np.array(
    [[[808, 395]], [[870, 397]], [[1074, 418]], [[1417, 454]], [[1750, 498]], [[1825, 511]], 
[[986, 431]], [[860, 451]], [[803, 457]], [[740, 468]], 
[[1356, 471]], [[1260, 495]], [[1203, 507]], [[1145, 528]]], np.float32)

objp_cam3 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam3 = np.array(
    [[[838, 615]], [[867, 626]], [[1016, 680]], [[1332, 784]], [[1767, 925]], [[1888, 953]], 
[[860, 680]], [[636, 676]], [[544, 675]], [[456, 673]], 
[[1135, 805]], [[833, 823]], [[692, 827]], [[532, 826]]], np.float32)

objp_cam4 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0],
     [0, 1400, 0], [1500, 1400, 0]], np.float32)

corners_cam4 = np.array(
    [[[1556, 498]], [[1534, 512]], [[1454, 567]], [[1245, 694]], [[780, 965]], [[602, 1061]], 
[[1292, 565]], [[1105, 567]], [[1036, 566]], [[977, 564]], 
[[1012, 684]], [[795, 668]], [[722, 663]], [[661, 660]], 
[[798, 518]], [[16, 701]]], np.float32)


save_objp = [objp_cam1, objp_cam2, objp_cam3, objp_cam4]
save_corners = [corners_cam1, corners_cam2, corners_cam3, corners_cam4]

np.save(matrix_save + 'objp.npy', save_objp)
np.save(matrix_save + 'corners.npy', save_corners)