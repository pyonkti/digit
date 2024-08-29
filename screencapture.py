import re
import os
import math
from matplotlib import pyplot as plt
from digit_interface import Digit
import cv2
import numpy as np
from datetime import datetime

def process_continuous_frames(d):
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()
    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detected_lines_{timestamp}.png")
            frame = d.get_frame()
            cv2.imwrite(output_path,frame)
            cv2.imshow("Detected Lines (in red)",frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


d = Digit("D20790") # Unique serial number
d.connect()
process_continuous_frames(d)
d.disconnect()
    

