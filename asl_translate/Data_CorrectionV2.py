#Quick AI fix for frozen/dropped frames in data MODIFIED FOR Data_collectionV2. Checks for duplicate frames and also enables the anchoring as added in V2 of data collection.

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

# --- 1. SCAN FOR CORRUPTED/FROZEN DATA ---
print(f"Scanning {DATA_PATH} for frozen frame drops...")
bad_sequences = []

for filename in os.listdir(DATA_PATH):
    if filename.endswith(".npy"):
        filepath = os.path.join(DATA_PATH, filename)
        try:
            data = np.load(filepath)
            
            # Check 1: Did the whole file somehow end up as zeros?
            if np.all(data == 0):
                seq_num = int(filename.replace('.npy', ''))
                bad_sequences.append(seq_num)
                continue
                
            # Check 2: Are there any duplicated consecutive frames? (The Failsafe trigger)
            failsafe_triggered = False
            for i in range(1, data.shape[0]):
                # If the current frame is EXACTLY equal to the previous frame
                if np.array_equal(data[i], data[i-1]):
                    failsafe_triggered = True
                    break
                    
            if failsafe_triggered:
                seq_num = int(filename.replace('.npy', ''))
                bad_sequences.append(seq_num)
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if not bad_sequences:
    print("-" * 40)
    print(f"Success! No drops or frozen frames found in the {action} dataset.")
    sys.exit()

bad_sequences.sort()
print("-" * 40)
print(f"Corrupted/Frozen sequences identified: {bad_sequences}")
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
    
    # NEW: Apply the Anchor and Failsafe logic to the corrector as well
    anchor_coord = None
    last_good_keypoints = np.zeros(21 * 3)
    
    for frame_num in range(sequence_length):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        
        yolo_results = yolo_model(frame, verbose=False)
        boxes = yolo_results[0].boxes
        
        # Default to the failsafe
        keypoints = last_good_keypoints 
        
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
                    
                    # 1. ESTABLISH THE FRAME'S ANCHOR (The Wrist)
                    wrist = hand_landmarks[0]
                    # We need the wrist in global pixel coordinates first
                    wrist_global_x = x1 + int(wrist.x * crop_w)
                    wrist_global_y = y1 + int(wrist.y * crop_h)
                    
                    # Normalize the wrist to screen scale (0 to 1)
                    wrist_norm_x = wrist_global_x / frame_w
                    wrist_norm_y = wrist_global_y / frame_h
                    wrist_z = wrist.z # MediaPipe Z is already relative to the wrist, but we lock it here

                    for landmark in hand_landmarks:
                        # 2. GET CURRENT LANDMARK IN GLOBAL PIXELS
                        global_x = x1 + int(landmark.x * crop_w)
                        global_y = y1 + int(landmark.y * crop_h)
                        
                        # 3. NORMALIZE TO SCREEN SCALE (0 to 1)
                        raw_norm_x = global_x / frame_w
                        raw_norm_y = global_y / frame_h
                        
                        # 4. SUBTRACT THE ANCHOR (The Magic Fix)
                        # This makes the wrist ALWAYS (0,0,0). 
                        # If a fingertip is at X: 0.6 and the wrist is at X: 0.5, the saved value is 0.1
                        final_x = raw_norm_x - wrist_norm_x
                        final_y = raw_norm_y - wrist_norm_y
                        final_z = landmark.z - wrist_z 
                        
                        extracted_points.extend([final_x, final_y, final_z])
                        
                        # Visual feedback (We use the global pixels so it still draws on the screen properly)
                        cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                        
                    keypoints = np.array(extracted_points)
                    last_good_keypoints = keypoints

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
