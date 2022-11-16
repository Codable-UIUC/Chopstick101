import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
filename = "data/c.mp4"
cap = cv2.VideoCapture(filename)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  cnt = 0
  data = []
  while cap.isOpened():
    frame = []
    success, image = cap.read()
    if not success:
      print("Completed.")
      break
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      
      for i in (results.multi_hand_landmarks[0].landmark):
        frame.append([i.x, i.y, i.z])
        # # of frames x 21 landmarks x 3 properties (N, 21, 3)
      frame = np.array(frame)
      data.append(frame)

      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      cv2.imwrite(f'videoframe/new{cnt}.png', image)
      # out.write(image)
      cnt += 1
  outputfile = filename[5:-3] + "pkl"
  with open(outputfile,'wb') as f:
    pickle.dump(np.array(data), f)
cap.release()

