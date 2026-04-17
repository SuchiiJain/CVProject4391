#Helper which makes a series of frames a then gives them to the model
from collections import deque
import numpy as np

class FrameBuffer:
    #Current series_lenght is 16 -- can adjust to 8 if needed
    def __init__(self, series_length = 16):
        self.buffer = deque(maxlen=series_length)
        self.series_length = series_length

    def add_frame(self, keypoints):
        self.buffer.append(keypoints)
    
    #Check for a full series
    def is_full_series(self):
        return len(self.buffer) == self.series_length

    def get_series(self):
        return np.array(self.buffer)
    
    #No hand detected -- buffer stays still
    def pause(self):
        pass
    
