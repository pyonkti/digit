import os
from digit_interface import Digit
import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim # type: ignore
from utils.draw_lines import (
    draw_line_and_parallelogram,
    remove_vertical_lines,
    lightglue_detection_area
)

class FIFOQueue:
    def __init__(self, size):
        self.queue = []
        self.size = size
    
    def enqueue(self, item):
        if len(self.queue) >= self.size:
            self.dequeue()  # Remove the oldest item to make space
        self.queue.append(item)
    
    def dequeue(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None  # or raise an exception
    
    def __str__(self):
        return str(self.queue)
    
    def __len__(self):
        return len(self.queue)
    
    def clear(self):
        self.queue = []

def compare_images(imageA, imageB):
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"
    
    ssim_value, _ = ssim(imageA, imageB, full=True)

    return ssim_value


def count_edge_pixels_in_parallelogram(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], 255)
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    total_pixels = cv2.countNonZero(mask)
    return float(edge_pixels/total_pixels)

def process_continuous_frames():
    """
    Continuously captures and processes frames from the DIGIT device.
    """

    gaussian = 23
    median = 5
    canny_threshold1=20
    canny_threshold2=70
    hough_rate = 44
    #line_threshold = 10
    break_rate = 35
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength= 117
    maxLineGap= 51
    parallelogram_points = None
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    background_frame = cv2.imread('/root/digit/edge_detection_autoencoder/dataset/empty/detected_lines_20240820_141634.png')
    blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian, gaussian), 0), median)
    while True:
        try:
            temp_hough = hough_rate

            frame = cv2.imread('/root/digit/edge_detection_autoencoder/dataset/images/detected_lines_20240806_172602.png')
            height, width, channels = frame.shape
            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and help in edge detection
            blurred_image = cv2.GaussianBlur(grey_image, (gaussian, gaussian), 0)
            blurred_image = cv2.medianBlur(blurred_image,median)
            if blurred_base_frame is not None:  
                ssim_value = compare_images(blurred_base_frame,blurred_image)
                blurred_image = cv2.absdiff(blurred_image, blurred_base_frame)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_image = clahe.apply(blurred_image)

            if parallelogram_points is None and ssim_value > 0.92:
                edges = np.zeros((height, width, channels), dtype=np.uint8)
                tiled_layout = np.zeros((height, width * 3, channels), dtype=np.uint8)
                # Place images into the layout
                tiled_layout[0:height, 0:width] = frame
                tiled_layout[0:height, width:width*2] = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
                tiled_layout[0:height, width*2:width*3] = edges

                cv2.imshow("Detected Lines (in red)",tiled_layout)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            # Canny Edge Detection
            edges = cv2.Canny(image=enhanced_image, threshold1=canny_threshold1, threshold2=canny_threshold2) # Canny Edge Detection

            # First attempt to find lines with initial rate
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)
            
            # Adjust rate until lines are found, adhering to break_rate limit
            if len(lines) == 0:
                while temp_hough > break_rate:
                    temp_hough -= threshold_increment
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
                    lines = remove_vertical_lines(lines)

                    if len(lines) > 0:
                        parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                        lightGlue_area = lightglue_detection_area(lines,frame)
                        break
            else:
                parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                lightGlue_area = lightglue_detection_area(lines,frame)

            tiled_layout = np.zeros((height, width * 3, channels), dtype=np.uint8)

            # Place images into the layout
            tiled_layout[0:height, 0:width] = frame
            tiled_layout[0:height, width:width*2] = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
            tiled_layout[0:height, width*2:width*3] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Detected Lines (in red)",tiled_layout)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    process_continuous_frames()