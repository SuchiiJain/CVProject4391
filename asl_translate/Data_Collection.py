import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import time

# ---CONFIG---
# Change this for every letter
action = 'A' 
# Videos of letter in folder (Don't Change this, for simplicities sake)
no_sequences = 30 
# How many frames per video (Don't Change this, due to how LSTM's Work)
sequence_length = 16 

# Folder location
DATA_PATH = os.path.join('ASL_Dataset')
try: 
    os.makedirs(os.path.join(DATA_PATH, action))
except OSError:
    pass

# Turns on Yolo 
yolo_model = YOLO('yolo26n.pt') 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # Only use one hand for signing
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Loop through video amount
for sequence in range(no_sequences):
    # Loop through frames (16 frames per video)
    window = [] # This will hold the 16 arrays of 63 ((x,y,z) * 16 frames) coordinates
    
    for frame_num in range(sequence_length):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        yolo_results = yolo_model(frame, verbose=False)
        boxes = yolo_results[0].boxes
        
        # Default empty array if no hand is found to prevent crashing
        keypoints = np.zeros(21 * 3) 
        
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                crop_h, crop_w, _ = cropped_frame.shape
                rgb_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                mp_results = detector.detect(mp_image)

                if mp_results.hand_landmarks:
                    # Grab detected hand and apply landmarks to it
                    hand_landmarks = mp_results.hand_landmarks[0] 
                    extracted_points = []
                    
                    for landmark in hand_landmarks:
                        # Translate local crop coord back to global pixel coord
                        global_x = x1 + int(landmark.x * crop_w)
                        global_y = y1 + int(landmark.y * crop_h)
                        
                        # Normalize by screen resolution (works for any screen res size) (returns a float between 0 and 1)
                        norm_x = global_x / frame_w
                        norm_y = global_y / frame_h
                        
                        # MediaPipe's Z is already relative, keep as is
                        extracted_points.extend([norm_x, norm_y, landmark.z])
                        
                        # Visual feedback
                        cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                        
                    keypoints = np.array(extracted_points)

        # Append the 63 coordinates to our current video sequence
        window.append(keypoints)

        # User's Cues
        if frame_num == 0: 
            cv2.putText(frame, 'GET READY...', (120, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(2000) # 2 second pause between videos to reset hand
        else: 
            cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(10) # 10ms wait for a smooth 16-frame capture
            
    # Saves to file location
    # Once the loop hits 16 frames, save the 2D array as a .npy file (apparently .npy files are better than our original .csv)
    npy_path = os.path.join(DATA_PATH, action, str(sequence))
    np.save(npy_path, np.array(window))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
