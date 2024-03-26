import re
import math
from matplotlib import pyplot as plt
from digit_interface import Digit
from digit_interface import DigitHandler
import cv2
import numpy as np

def process_continuous_frames(d):
    """
    Continuously captures and processes frames from the DIGIT device.
    """
    try:
        while True:
            frame = d.get_frame()  # Or False, based on your needs

            if not isinstance(frame, np.ndarray):
                print("Error: Frame is not a valid numpy array.")
                exit()

            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Check if the image is loaded successfully
            if not isinstance(grey_image, np.ndarray):
                print("Error: Conversion to grayscale failed.")
                exit()

            # Apply GaussianBlur to reduce noise and help in edge detection
            blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
            if not isinstance(blurred_image, np.ndarray):
                print("Error: Gaussian Blur application failed.")
                exit()
            # Canny Edge Detection
            edges = cv2.Canny(image=blurred_image, threshold1=0, threshold2=10) # Canny Edge Detection

            rate = 100
            theta_min = np.pi / 180 * 10  
            theta_max = np.pi / 180 * 170
            
            
            cv2.imshow("Detected Lines (in red)", blurred_image)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


d = Digit("D20782") # Unique serial number
d.connect()
process_continuous_frames(d)
d.disconnect()