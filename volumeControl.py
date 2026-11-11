import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg 
import subprocess


def volume_up():
    subprocess.run([
        "osascript", "-e",
        # "tell application \"System Events\" to key code 144"
        "set volume output volume (output volume of (get volume settings) + 6.25)"
    ], check=True)

def volume_down():
    subprocess.run([
        "osascript", "-e",
        "set volume output volume (output volume of (get volume settings) - 6.25)"
    ], check=True)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
frameReduction = 20
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence=0.7, max_num_hands = 1)
wrappedUp = 0
screen_width,screen_height = pg.size()
cap = cv2.VideoCapture(1) # 0 if default cam is inbuilt and 1 if its external
h= 0
w=0
thumbStraightness = 0
thumb = 'No thumbs up/down detected'

def getCoords(indexVal):
    return int(landmarks[indexVal].x*w), int(landmarks[indexVal].y*h)

if not cap.isOpened():
    print('could not access the camera')
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print('cant receive frame')
        break
    frame = cv2.flip(frame, 1)
    h, w, _  = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for land in hand.landmark:
                # print(f"x: {land.x} ,y: {land.y}, z: {land.z} " )
                pass
    # print(results)
    cv2.rectangle(frame, (frameReduction,frameReduction), (w-frameReduction, h-frameReduction), (255,0,0), 2)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            thumb = 'No thumbs up/down detected'
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # fingers = [(8,5),(12,9),(16,13),(20,17)]
            fingers = [(8,6),(12,10),(16,14),(20,18)]
            wrappedUp = 0
            for tip, base in fingers:
                # if getCoords(tip)[0] > getCoords(base)[0]:
                if getCoords(tip)[1] > getCoords(base)[1]:
                    wrappedUp += 1
            thumbStraightness = abs(getCoords(4)[0] - getCoords(1)[0])
            if wrappedUp >= 3:
               
                if abs(getCoords(4)[0] - getCoords(1)[0]) < 110:
                    # if getCoords(2)[1] > getCoords(5)[1] and getCoords(5)[1]<getCoords(17)[1]: 
                    if getCoords(4)[1] < getCoords(2)[1]-10:
                        print('volume should be going up')
                        thumb = 'Volume is going up'
                        volume_up()
                    # elif getCoords(2)[1] > getCoords(5)[1] and getCoords(5)[1]>getCoords(17)[1]:
                    elif getCoords(4)[1] > getCoords(2)[1]+10:
                        print('volume should be going down')
                        thumb = 'Volume is going down'
                        volume_down()

    cv2.putText(frame, f"wrappedUp: {wrappedUp}", (40,50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"thumb straight: {thumbStraightness}", (40,80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"{thumb}.", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    

    cv2.imshow('window', frame)


    
    # wait for 1 milisec and check if user presses q, stop the camera. 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

    


    
#thumbs up increase volume

            
# Activation command - source .venv/bin/activate
# check mediapipe verion - pip show mediapipe
# mediapipe - Name: mediapipe
# Version: 0.10.14
# Summary: MediaPipe is the simplest way for researchers and developers to build world-class ML solutions and applications for mobile, edge, cloud and the web.
# Home-page: https://github.com/google/mediapipe
# Author: The MediaPipe Authors
# Author-email: mediapipe@google.com
# License: Apache 2.0
# Location: /Users/adityapurwar/Desktop/Bhumis_coding/Python/OpenCV_Coding/.venv/lib/python3.11/site-packages
# Requires: absl-py, attrs, flatbuffers, jax, jaxlib, matplotlib, numpy, opencv-contrib-python, protobuf, sounddevice
# Required-by:

