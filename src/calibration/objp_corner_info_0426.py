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

matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0426/information/'
cam_num = 4


objp_cam1 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam1 = np.array(
    [[[343, 718]], [[408, 723]], [[753, 734]], [[1200, 738]], [[1544, 728]], [[1608, 725]],
     [[722, 762]], [[665, 802]], [[636, 823]], [[615, 846]],
     [[1233, 764]], [[1282, 802]], [[1312, 825]], [[1338, 852]]], np.float32)


objp_cam2 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam2 = np.array(
    [[[887, 460]], [[932, 467]], [[1148, 492]], [[1492, 531]], [[1822, 575]], [[1892, 586]],
     [[1066, 501]], [[924, 518]], [[868, 526]], [[807, 535]],
     [[1434, 548]], [[1332, 574]], [[1275, 588]], [[1217, 610]]], np.float32)


objp_cam3 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam3 = np.array(
    [[[854, 480]], [[887, 490]], [[1038, 535]], [[1355, 625]], [[1805, 757]], [[1917, 794]],
     [[882, 545]], [[661, 558]], [[568, 565]], [[482, 572]],
     [[1165, 653]], [[866, 686]], [[726, 703]], [[568, 718]]], np.float32)


objp_cam4 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0],
     [0, 1400, 0], [1500, 1400, 0]], np.float32)

corners_cam4 = np.array(
    [[[1576, 606]], [[1557, 614]], [[1461, 654]], [[1231, 747]], [[735, 915]], [[528, 975]],
     [[1304, 624]], [[1115, 587]], [[1053, 573]], [[997, 564]],
     [[1007, 693]], [[808, 632]], [[726, 611]], [[667, 592]],
     [[832, 481]], [[26, 505]]], np.float32)


save_objp = [objp_cam1, objp_cam2, objp_cam3, objp_cam4]
save_corners = [corners_cam1, corners_cam2, corners_cam3, corners_cam4]

np.save(matrix_save + 'objp.npy', save_objp)
np.save(matrix_save + 'corners.npy', save_corners)