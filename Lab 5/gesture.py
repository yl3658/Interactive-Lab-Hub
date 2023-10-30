import cv2
import mediapipe as mp
import math
import hand_unistroke
import time
import numpy as np


def vector_2d_angle(v1,v2): # 求出v1,v2兩條向量的夾角
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_ = math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 100000.
    return angle_

def hand_angle(hand_):
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    #print(angle_list)
    return angle_list

def hand_gesture(angle_list):
    gesture_str = 'none'
    if 100000. not in angle_list:
        if (angle_list[0]>20 and angle_list[0]<70) and (angle_list[1]>70 and angle_list[1]<120) and (angle_list[2]>140 and angle_list[2]<180) and (angle_list[3]>140 and angle_list[3]<180) and (angle_list[4]>130 and angle_list[4]<180):
            gesture_str = "latte"
    return gesture_str

def detect(artname):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    # 開啟視訊鏡頭讀取器
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
    while True:
        # 偵測影像中的手部
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing = mp.solutions.drawing_utils

        if results.multi_hand_landmarks:
            
            # drawing rectangle needed
            for hand_landmarks in results.multi_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                
            
            for hand_landmarks in results.multi_hand_landmarks:
                keypoint_pos = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    keypoint_pos.append((x,y))
                if keypoint_pos:
                    # 得到各手指的夾角資訊
                    angle_list = hand_angle(keypoint_pos)
                    # 根據角度判斷此手勢是否為拉花
                    gesture_str = hand_gesture(angle_list)
                    if gesture_str == "latte":
                        # draw rectangle
                        cv2.putText(frame, "Latte Art", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
                        img_counter = 0
                        start_time = time.time()
                        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')         # 設定影片的格式
                        out = cv2.VideoWriter('capture_output.mp4', fourcc, 20.0, (width,  height))  # 產生空的影片
                        tracker = hand_unistroke.handTracker()
                        finger_pos_list = []
                        recognizer = hand_unistroke.Recognizer()
                        track_quit = 0
                        
                        while track_quit == 0:
                            success,image = cap.read()
                            image = cv2.flip(image,1)
                            image = tracker.handsFinder(image)
                            lmList = tracker.positionFinder(image)
                            lift = 10
                            if len(lmList) != 0:
                                index_finger_pos = lmList[8]
                                finger_pos_list.append((index_finger_pos[1],index_finger_pos[2]))
                                cv2.ellipse(image, (finger_pos_list[0][0], finger_pos_list[0][1]), (200, 70), 0, 0, 360, (114,152,189), -1)
                                if len(finger_pos_list) >= 2:
                                    in_range = 0
                                    while in_range < len(finger_pos_list)-1:
                                        if finger_pos_list[in_range][1] - finger_pos_list[in_range+1][1] > 6:
                                            lift = in_range
                                        cv2.line(image, finger_pos_list[in_range],finger_pos_list[in_range+1], (0,0,0), 3) 
                                        in_range+=1
                            out.write(image)
                            cv2.imshow("track_frame",image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                track_quit=1
                                break
                        cv2.ellipse(image, (finger_pos_list[0][0], finger_pos_list[0][1]), (200, 70), 0, 0, 360, (114,152,189), -1)
                        cv2.ellipse(image, (finger_pos_list[0][0], finger_pos_list[0][1]), (max(lift*0.02,30), max(lift*0.04,50)), 270-30, 0, 180, (255,255,255), -1)
                        cv2.ellipse(image, (finger_pos_list[0][0], finger_pos_list[0][1]), (max(lift*0.02,30), max(lift*0.04,50)), 270+30, 0, 180, (255,255,255), -1)
                        cv2.drawContours(image, [np.array( [(finger_pos_list[0][0], finger_pos_list[0][1]+32), (finger_pos_list[0][0], finger_pos_list[0][1]-32), (finger_pos_list[len(finger_pos_list)-1][0]+50,finger_pos_list[0][1])])], 0, (255,255,255), -1)
                        cv2.putText(image, str(recognizer.get_gesture(finger_pos_list,artname)), (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 1, cv2.LINE_AA)
                        cv2.imshow('result',image)
                        print(finger_pos_list)
                        print(str(recognizer.get_gesture(finger_pos_list,artname)))
                        out.release()
                        cv2.destroyWindow('track_frame') 
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow('MediaPipe Hands')
    cv2.destroyWindow('track_frame')
    cap.release()
if __name__ == '__main__':
    detect()