#Quick AI fix for zeroes in data

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import os
import sys
import time
from frame_buffer import FrameBuffer

# --- CONFIG ---
action = input("Enter the letter you want to check and fix (e.g., A): ").upper()
DATA_PATH = os.path.join('ASL_Dataset', action)
sequence_length = 16 

if not os.path.exists(DATA_PATH):
    print(f"Directory {DATA_PATH} does not exist.")
    sys.exit()

# --- 1. SCAN FOR CORRUPTED DATA ---
print(f"Scanning {DATA_PATH} for corrupted arrays...")
bad_sequences = []

for filename in os.listdir(DATA_PATH):
    if filename.endswith(".npy"):
        filepath = os.path.join(DATA_PATH, filename)
        try:
            data = np.load(filepath)
            # If any zeros are found, grab the sequence number from the filename
            if np.any(data == 0):
                seq_num = int(filename.replace('.npy', ''))
                bad_sequences.append(seq_num)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if not bad_sequences:
    print("-" * 40)
    print(f"Success! No zeros found in the {action} dataset. You are good to go.")
    sys.exit()

bad_sequences.sort()
print("-" * 40)
print(f"Corrupted sequences identified: {bad_sequences}")
print(f"Total files to re-record: {len(bad_sequences)}")
input("Press ENTER to start the camera and patch these files...")

# --- 2. RE-RECORDING SETUP ---
yolo_model = YOLO('yolo26n.pt') 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, 
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def cleanup():
    try:
        detector.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()

# --- 3. SURGICAL OVERWRITE LOOP ---
for sequence in bad_sequences:
    window = FrameBuffer(series_length=sequence_length)
    
    for frame_num in range(sequence_length):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        yolo_results = yolo_model(frame, verbose=False)
        boxes = yolo_results[0].boxes
        
        keypoints = np.zeros(21 * 3) 
        
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                rgb_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                mp_results = detector.detect(mp_image)

                if mp_results.hand_landmarks:
                    hand_landmarks = mp_results.hand_landmarks[0] 
                    extracted_points = []
                    
                    for landmark in hand_landmarks:
                        global_x = x1 + int(landmark.x * cropped_frame.shape[1])
                        global_y = y1 + int(landmark.y * cropped_frame.shape[0])
                        norm_x = global_x / frame_w
                        norm_y = global_y / frame_h
                        
                        extracted_points.extend([norm_x, norm_y, landmark.z])
                        cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                        
                    keypoints = np.array(extracted_points)

        window.add_frame(keypoints)

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        status_text = f"FIXING: {action} | Overwriting File: {sequence}.npy | ESC to Quit"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)

        if frame_num == 0: 
            cv2.putText(frame, f'RE-RECORDING FILE {sequence}...', (80, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('Data Correction', frame)
            
            for _ in range(20): 
                if cv2.waitKey(100) & 0xFF == 27: 
                    print("Exiting during countdown...")
                    cleanup()
                    sys.exit() 
        else: 
            cv2.imshow('Data Correction', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting during capture...")
                cleanup()
                sys.exit()
            
    # Save directly over the corrupted file
    npy_path = os.path.join(DATA_PATH, str(sequence))
    np.save(npy_path, window.get_series())
    print(f"Successfully overwrote {sequence}.npy")

cleanup()
print("\nAll targeted sequences have been patched!")
