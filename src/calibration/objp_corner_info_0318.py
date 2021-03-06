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

matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0318/information/'
cam_num = 4


objp_cam1 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam1 = np.array(
    [[[611, 431]], [[683, 436]], [[1035, 436]], [[1474, 443]], [[1794, 463]], [[1854, 467]],
     [[1002, 460]], [[952, 500]], [[917, 526]], [[891, 546]],
     [[1507, 472]], [[1554, 514]], [[1581, 537]], [[1607, 571]]], np.float32)


objp_cam2 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam2 = np.array(
    [[[664, 557]], [[704, 561]], [[931, 576]], [[1284, 598]], [[1637, 621]], [[1714, 624]],
     [[834, 592]], [[710, 612]], [[650, 620]], [[594, 633]],
     [[1222, 623]], [[1127, 652]], [[1076, 668]], [[1012, 690]]], np.float32)


objp_cam3 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam3 = np.array(
    [[[802, 464]], [[828, 472]], [[977, 523]], [[1304, 621]], [[1761, 775]], [[1882, 815]],
     [[823, 531]], [[605, 538]], [[508, 545]], [[420, 543]],
     [[1108, 644]], [[805, 673]], [[660, 685]], [[500, 698]]], np.float32)


objp_cam4 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0],
     [0, 1400, 0], [1500, 1400, 0]], np.float32)

corners_cam4 = np.array(
    [[[1570, 543]], [[1555, 554]], [[1474, 606]], [[1276, 725]], [[808, 974]], [[601, 1067]],
     [[1306, 585]], [[1108, 568]], [[1044, 560]], [[978, 552]],
     [[1034, 683]], [[788, 644]], [[718, 628]], [[651, 617]],
     [[782, 483]], [[3, 581]]], np.float32)


save_objp = [objp_cam1, objp_cam2, objp_cam3, objp_cam4]
save_corners = [corners_cam1, corners_cam2, corners_cam3, corners_cam4]

np.save(matrix_save + 'objp.npy', save_objp)
np.save(matrix_save + 'corners.npy', save_corners)