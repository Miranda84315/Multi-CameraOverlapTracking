import cv2
from os import listdir
from os.path import isfile, join

data_path = ['datasets\\calibration\\temp\\c1\\', 'datasets\\calibration\\temp\\c2\\',
             'datasets\\calibration\\temp\\c3\\', 'datasets\\calibration\\temp\\c4\\',
             'datasets\\calibration_world\\test\\']
data_path_save = ['datasets\\resize\\calibration\\c1\\', 'datasets\\resize\\calibration\\c2\\',
                  'datasets\\resize\\calibration\\c3\\', 'datasets/resize/calibration/c4/',
                  'datasets/resize/calibration/world/']

for file_name in range(1, 1141):
    for c_num in range(0, 4):
        path = data_path[c_num] + str(file_name) + '.jpg'
        save_path = data_path_save[c_num] + str(file_name) + '.jpg'
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 360))
        cv2.imwrite(save_path, img)

    print(file_name)

cv2.destroyAllWindows()
for i in range(0, len(data_path)):
    onlyfiles = [f for f in listdir(data_path[i]) if isfile(join(data_path[i], f))]

    for file_name in onlyfiles:
        path = data_path[i] + file_name
        save_path = data_path_save[i] + file_name
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 360))
        cv2.imwrite(save_path, img)
        print(path)

cv2.destroyAllWindows()

