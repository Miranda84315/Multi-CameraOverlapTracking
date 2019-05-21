'''
This is use for calucate total frame from player01 to player52 and each track
'''
import cv2

video_root = 'D:/Code/MultiCamOverlap/dataset/videos/Player'
players = []
for i in range(1, 10):
    players.append('0' + str(i))
for i in range(10, 53):
    players.append(str(i))

for player in range(0, 52):
    player_frame = []
    for track in range(0, 8):
        video_dir = video_root + players[player] + '/track' + str(track + 1)
        total_frame = []
        for cam in range(0, 4):
            video_name = video_dir + '/cam' + str(cam + 1) + '.avi'
            cap = cv2.VideoCapture(video_name)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frame.append(frame_num)
        #print('===== Player ' + str(player + 1) + ', track' + str(track + 1) + '===== -> ' + str(min(total_frame)))
        player_frame.append(min(total_frame))
    #print('===== Player' + str(player + 1) + ' =====')
    print(player_frame[0], player_frame[1], player_frame[2], player_frame[3], player_frame[4], player_frame[5], player_frame[6], player_frame[7])

