import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import time
import sys
from frame_buffer import FrameBuffer # Import the helper for series of frames

# ---CONFIG---
# Change this for every letter
action = 'A' 
# Videos of letter in folder (Don't Change this, for simplicities sake)
num_sequences = 30 
# How many frames per video (Don't Change this, due to how LSTM's Work)
sequence_length = 16 

print("Team Members: 1=Suchi_Jain, 2=Lauren_Anderson, 3=Victor_Runyan, 4=Nickolas_Ackley")
user_id = int(input("Enter your User Number (1-4): "))

# The same user will only change their specific set of videos (no overriding other user's videos
start_sequence = (user_id - 1) * num_sequences
end_sequence = user_id * num_sequences

print(f"\n User {user_id} is recording videos {start_sequence} through {end_sequence - 1} for the letter {action}.")

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

# Helper designed to exit safely without errors 
def cleanup():
    try:
        detector.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    

# Loop through video amount
for sequence in range(start_sequence, end_sequence): #New videos per user that go to the same location
    # Loop through frames (16 frames per video)
    window = FrameBuffer(series_length=sequence_length) # Create a new buffer for each video sequence
    
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

       
        window.add_frame(keypoints)

      # User Cues
        # Draw a background bar for the status text
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        
        status_text = f"Action: {action} | Video: {sequence} | ESC to Quit"
        cv2.putText(frame, status_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if frame_num == 0: 
            # 2. THE RESPONSIVE PAUSE
            # Instead of one big 2-second wait, we check for ESC 20 times.
            cv2.putText(frame, f'GET READY FOR {action}...', (120, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            
            for _ in range(20): # 20 iterations * 100ms = 2 seconds
                if cv2.waitKey(100) & 0xFF == 27: 
                    print("Exiting during countdown...")
                    cleanup()
                    sys.exit() # Kills the script immediately
        else: 
            # 3. NORMAL CAPTURE
            cv2.imshow('Data Collection', frame)
            # Check for ESC during every single frame capture
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting during capture...")
                cleanup()
                sys.exit()
            
    # Saves to file location
    # Once the loop hits 16 frames, save the 2D array as a .npy file (apparently .npy files are better than our original .csv)
    npy_path = os.path.join(DATA_PATH, action, str(sequence))
    np.save(npy_path, window.get_series())

cleanup()
