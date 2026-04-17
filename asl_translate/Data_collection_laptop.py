import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import sys
from frame_buffer import FrameBuffer

# ---CONFIG---
# Change this for every letter
action = 'A'
# Videos of letter in folder (Don't Change this, for simplicities sake)
num_sequences = 30
# How many frames per video (Don't Change this, due to how LSTM's Work)
sequence_length = 16

print("Team Members: 1=Suchi_Jain, 2=Lauren_Anderson, 3=Victor_Runyan, 4=Nickolas_Ackley")
user_id = int(input("Enter your User Number (1-4): "))

# The same user will only change their specific set of videos (no overriding other user's videos)
start_sequence = (user_id - 1) * num_sequences
end_sequence = user_id * num_sequences

print(f"\nUser {user_id} is recording videos {start_sequence} through {end_sequence - 1} for the letter {action}.")

# Folder location
DATA_PATH = os.path.join('ASL_Dataset')
try:
    os.makedirs(os.path.join(DATA_PATH, action))
except OSError:
    pass

# MediaPipe only -- no YOLO on laptop
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Helper designed to exit safely without errors
def cleanup():
    try:
        detector.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()

# Loop through video amount
for sequence in range(start_sequence, end_sequence):

    # Fresh buffer for every new video
    window = FrameBuffer(series_length=sequence_length)

    # --- COUNTDOWN: keep camera live while user gets ready ---
    for _ in range(20):  # 20 iterations * 100ms = 2 seconds
        success, live_frame = cap.read()
        live_frame = cv2.flip(live_frame, 1)
        frame_h, frame_w, _ = live_frame.shape
        cv2.rectangle(live_frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        cv2.putText(live_frame, f"Action: {action} | Video: {sequence} | ESC to Quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(live_frame, f'GET READY FOR {action}...', (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Data Collection', live_frame)
        if cv2.waitKey(100) & 0xFF == 27:
            print("Exiting during countdown...")
            cleanup()
            sys.exit()

    # --- RECORD: capture exactly 16 frames ---
    for frame_num in range(sequence_length):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        # Default empty array if no hand is found to prevent crashing
        keypoints = np.zeros(21 * 3)

        # Pass the full frame directly to MediaPipe (no YOLO crop needed)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = detector.detect(mp_image)

        if mp_results.hand_landmarks:
            hand_landmarks = mp_results.hand_landmarks[0]
            extracted_points = []

            for landmark in hand_landmarks:
                norm_x = landmark.x
                norm_y = landmark.y
                extracted_points.extend([norm_x, norm_y, landmark.z])

                # Visual feedback
                cv2.circle(frame, (int(landmark.x * frame_w), int(landmark.y * frame_h)), 5, (0, 255, 0), -1)

            keypoints = np.array(extracted_points)

        window.add_frame(keypoints)

        # User Cues
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Action: {action} | Video: {sequence} | ESC to Quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            print("Exiting during capture...")
            cleanup()
            sys.exit()

    # Save the completed 16-frame series as a .npy file
    npy_path = os.path.join(DATA_PATH, action, str(sequence))
    np.save(npy_path, window.get_series())

cleanup()