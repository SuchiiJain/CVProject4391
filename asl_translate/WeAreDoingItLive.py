# Another AI Boilerplate!! NEEDS SPECIFIC WORKS
# Gemini Pro my Beloved

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import torch
from ASL_Model import ASLSequenceInterpreter # Import your brain architecture

# --- CONFIGURATION ---
# These MUST match exactly what you used in your training script!
actions = ['A', 'B', 'C'] 
sequence_length = 16

# --- 1. LOAD THE AI MODELS ---
print("Loading YOLO...")
yolo_model = YOLO('yolo26n.pt') 

print("Loading MediaPipe...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, 
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

print("Loading the LSTM Brain...")
# Force PyTorch to use the Jetson's GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLSequenceInterpreter(num_classes=len(actions)).to(device)

# Load the weights you saved during training
model.load_state_dict(torch.load('best_asl_model.pth', map_location=device))
model.eval() # CRITICAL: Puts the model in "prediction" mode, not training mode

# --- 2. LIVE VARIABLES ---
sequence = [] # This is your rolling window
current_prediction = "Waiting for data..."

# Turn on the webcam!
cap = cv2.VideoCapture(0)

print("We are doing it live! Press 'Esc' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # 1. Ask YOLO where the hand is
    yolo_results = yolo_model(frame, verbose=False)
    boxes = yolo_results[0].boxes
    
    keypoints = None # Reset keypoints for this frame
    
    # 2. If YOLO finds a hand, ask MediaPipe for the skeleton
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
                hand_landmarks = mp_results.hand_landmarks[0] 
                extracted_points = []
                
                for landmark in hand_landmarks:
                    global_x = x1 + int(landmark.x * crop_w)
                    global_y = y1 + int(landmark.y * crop_h)
                    norm_x = global_x / frame_w
                    norm_y = global_y / frame_h
                    extracted_points.extend([norm_x, norm_y, landmark.z])
                    
                    cv2.circle(frame, (global_x, global_y), 5, (0, 255, 0), -1)
                    
                keypoints = np.array(extracted_points)

    # --- THE FALLBACK REDUNDANCY ---
    # If the AI couldn't find a hand in this exact frame, handle it safely
    if keypoints is None:
        if len(sequence) > 0:
            # If we already have some data, just copy the previous frame's data 
            # to prevent the LSTM from experiencing a violent "jump" to zeros
            keypoints = sequence[-1] 
        else:
            # If it's the very first frame and nothing is there, use zeros
            keypoints = np.zeros(21 * 3)

    # Append to our rolling window
    sequence.append(keypoints)
    
    # Keep the window at exactly 16 frames by popping the oldest one off the front
    sequence = sequence[-16:]

    # --- 3. MAKE A PREDICTION ---
    # Only try to guess if we have a full 16 frames in the memory bank
    if len(sequence) == 16:
        # Convert our list of numpy arrays into a PyTorch Tensor
        # Add a dimension at the front because the model expects (Batch, Sequence, Features)
        input_data = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)
        
        # Don't track gradients (saves memory and speed during live inference)
        with torch.no_grad(): 
            res = model(input_data)
            
            # Find the index of the highest probability
            predicted_index = torch.argmax(res).item()
            
            # Map that index back to the actual letter (e.g., 0 -> 'A')
            current_prediction = actions[predicted_index]

    # --- 4. DISPLAY THE RESULT ---
    # Draw a black box behind the text so it's easy to read
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Prediction: {current_prediction}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('ASL Live Translation', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
