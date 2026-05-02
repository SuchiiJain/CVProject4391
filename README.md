# Real-Time American Sign Language Translation Via YOLO26 and MediaPipe-Assisted Sequence

## Teammates:
- Lauren Anderson
- Suchi Jain
- Victor Runyan
- Nicholas Ackley

## Overview: 
Our project builds a real-time American Sign Language (ASL) interpreter that converts hand gestures into letters, helping reduce the communication barrier between Deaf or Hard-of-Hearing individuals and people who do not know ASL.

The system uses a multi-stage machine learning pipeline to accurately recognize ASL gestures from live video input. First, YOLO26 is used to detect the user’s hand in each frame and crop out unnecessary background noise. This helps isolate the gesture and improves recognition accuracy.

The cropped frames are then processed using MediaPipe Hands, which extracts a 21-point skeletal representation of the hand. These landmark points capture the structure and movement of the hand and serve as the input features for gesture recognition.

Finally, the sequence of hand landmarks is passed into a Long Short-Term Memory (LSTM) neural network, which analyzes the temporal motion of the hand to classify the ASL letter being performed. The model supports recognition of all ASL alphabet letters, including dynamic gestures such as J and Z.

The output is the predicted letter corresponding to the detected gesture, enabling real-time translation of ASL hand signs into text natively via hardware acceleration.

---

## Jetson Orin Edge Deployment Setup

**Requirements:** Jetson Orin Nano/NX (JetPack 6.1), Python 3.10, native Ubuntu 22.04 terminal, a webcam.

**1. System Level Dependencies**
Install the necessary C++ image decoding libraries required by the ARM architecture:
```bash
sudo apt-get update
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
```

**2. Create and Activate the Virtual Enviroment**
Use standard Bash, not Windows PowerShell
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```
Never commit the venv/ folder. It is listed in .gitignore and should stay there.

**3. Install the hardware-Accelerated ML Pipeline**

Because we are deploying on NVIDIA ARM64 architecture, standard desktop pip packages will crash. Run the custom requirements file, which automatically bypasses security proxies to pull the Jetson AI Lab wheels and strict-locks the NumPy versions to prevent system conflicts.
```bash
pip3 install -r requirements.txt
```

**4. Download the MediaPipe landmarker Model**
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**5. Start the program**

After having trained off of data, best_asl_model.pth will appear as a file
```bash
python3 Live.py
```

**Verified Hardware Stack**

*Be aware that upgrading these packages will break the Jetson GPU pipeline*

OS Architecture: Ubuntu 22.04 (JetPack 6.1 L4T)

Python: 3.10

PyTorch: 2.5.0a0 (NVIDIA natively compiled)

TorchVision: 0.20.0 (Jetson AI Lab bypass)

NumPy: 1.26.4 (STRICT LOCK: NumPy 2.x breaks ARM PyTorch)

OpenCV: 4.9.0.80

MediaPipe: 0.10.5

Ultralytics (YOLO): 8.4.0
