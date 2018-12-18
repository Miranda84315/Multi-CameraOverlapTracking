import cv2
import numpy as np
import math
import os

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_name = ['datasets\\calibration_world\\c1-world.avi', 'datasets\\calibration_world\\c2-world.avi',
              'datasets\\calibration_world\\c3-world.avi', 'datasets\\calibration_world\\c4-world.avi',
              'datasets\\data1\\c1-data1.avi', 'datasets\\data1\\c2-data1.avi',
              'datasets\\data1\\c3-data1.avi', 'datasets\\data1\\c4-data1.avi',
              'datasets\\person\\c1-person.avi', 'datasets\\person\\c2-person.avi',
              'datasets\\person\\c3-person.avi', 'datasets\\person\\c4-person.avi',
              'datasets\\data2\\c1-data2.avi', 'datasets\\data2\\c2-data2.avi',
              'datasets\\data2\\c3-data2.avi', 'datasets\\data2\\c4-data2.avi'

              ]
images_path = ['datasets\\calibration_world\\c1_world\\', 'datasets\\calibration_world\\c2_world\\',
               'datasets\\calibration_world\\c3_world\\', 'datasets\\calibration_world\\c4_world\\',
               'datasets\\data1\\c1_data1\\', 'datasets\\data1\\c2_data1\\',
               'datasets\\data1\\c3_data1\\', 'datasets\\data1\\c4_data1\\',
               'datasets\\person\\c1_person\\', 'datasets\\person\\c2_person\\',
               'datasets\\person\\c3_person\\', 'datasets\\person\\c4_person\\',
               'datasets\\data2\\c1_data2\\', 'datasets\\data2\\c2_data2\\',
               'datasets\\data2\\c3_data2\\', 'datasets\\data2\\c4_data2\\'
               ]


for i in range(0, len(video_name)):

    print(video_name[i] + ' save frame starting -------------------')
    cap = cv2.VideoCapture(video_name[i])

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    count = 1

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            cv2.imwrite(images_path[i] + str(count) + '.jpg', frame)
            count += 1

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()






#   load image and save

for file_name in range(1, 1141):

    path1 = 'datasets\\data2\\c1_data2\\' + str(file_name) + '.jpg'
    path2 = 'datasets\\data2\\c2_data2\\' + str(file_name) + '.jpg'
    path3 = 'datasets\\data2\\c3_data2\\' + str(file_name) + '.jpg'
    path4 = 'datasets\\data2\\c4_data2\\' + str(file_name) + '.jpg'

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img3 = cv2.imread(path3)
    img4 = cv2.imread(path4)

    save_name = str(file_name + 1410) + '.jpg'

    cv2.imwrite('datasets\\all_data\\c1\\' + save_name, img1)
    cv2.imwrite('datasets\\all_data\\c2\\' + save_name, img2)
    cv2.imwrite('datasets\\all_data\\c3\\' + save_name, img3)
    cv2.imwrite('datasets\\all_data\\c4\\' + save_name, img4)

    print(file_name)



cv2.destroyAllWindows()

