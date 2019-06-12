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

matrix_save = 'D:/Code/MultiCamOverlap/dataset/calibration/0415_31/information/'
cam_num = 4


objp_cam1 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam1 = np.array(
    [[[404, 617]], [[466, 618]], [[811, 623]], [[1256, 624]], [[1597, 620]], [[1661, 616]], 
     [[781, 651]], [[727, 692]], [[700, 714]], [[674, 737]], 
     [[1291, 651]], [[1340, 692]], [[1365, 712]], [[1398, 741]]], np.float32)

objp_cam2 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam2 = np.array(
    [[[862, 405]], [[898, 408]], [[1121, 424]], [[1457, 451]], [[1793, 491]], [[1863, 500]], 
     [[1032, 437]], [[902, 458]], [[834, 470]], [[780, 478]], 
     [[1397, 472]], [[1301, 500]], [[1248, 513]], [[1182, 531]]], np.float32)

objp_cam3 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0]], np.float32)

corners_cam3 = np.array(
    [[[678, 496]], [[708, 501]], [[862, 525]], [[1181, 573]], [[1671, 654]], [[1805, 675]], 
     [[708, 556]], [[497, 598]], [[423, 615]], [[328, 637]], 
     [[996, 625]], [[704, 696]], [[570, 733]], [[427, 768]]], np.float32)

objp_cam4 = np.array(
    [[0, 0, 0], [90, 0, 0], [505, 0, 0], [995, 0, 0], [1410, 0, 0], [1500, 0, 0],
     [505, 175, 0], [505, 385, 0], [505, 470, 0], [505, 580, 0],
     [995, 175, 0], [995, 385, 0], [995, 470, 0], [995, 580, 0],
     [0, 1400, 0], [1500, 1400, 0]], np.float32)

corners_cam4 = np.array(
    [[[1553, 603]], [[1528, 613]], [[1438, 660]], [[1217, 761]], [[716, 952]], [[533, 1022]], 
     [[1275, 644]], [[1100, 626]], [[1031, 620]], [[973, 612]], 
     [[990, 725]], [[781, 690]], [[708, 675]], [[650, 663]], 
     [[795, 548]], [[12, 615]]], np.float32)


save_objp = [objp_cam1, objp_cam2, objp_cam3, objp_cam4]
save_corners = [corners_cam1, corners_cam2, corners_cam3, corners_cam4]

np.save(matrix_save + 'objp.npy', save_objp)
np.save(matrix_save + 'corners.npy', save_corners)