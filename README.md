# Real-Time American Sign Language Translation Via YOLO26 and MediaPipe-Assisted Sequence

# Teammates:
    - Lauren Anderson
    - Suchi Jain
    - Victor Runyan
    - Nicholas Ackley

# Overview: 

Our project builds a real-time American Sign Language (ASL) interpreter that converts hand gestures into letters, helping reduce the communication barrier between Deaf or Hard-of-Hearing individuals and people who do not know ASL.

The system uses a multi-stage machine learning pipeline to accurately recognize ASL gestures from video input. First, YOLO26 is used to detect the user’s hand in each frame and crop out unnecessary background noise. This helps isolate the gesture and improves recognition accuracy.

The cropped frames are then processed using MediaPipe Hands, which extracts a 21-point skeletal representation of the hand. These landmark points capture the structure and movement of the hand and serve as the input features for gesture recognition.

Finally, the sequence of hand landmarks is passed into a Long Short-Term Memory (LSTM) neural network, which analyzes the temporal motion of the hand to classify the ASL letter being performed. The model supports recognition of all ASL alphabet letters, including dynamic gestures such as J and Z.

The output is the predicted letter corresponding to the detected gesture, enabling real-time translation of ASL hand signs into text.

    