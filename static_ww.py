import os
from digit_interface import (
    Digit
)
import cv2
import torch
import numpy as np
import threading
import queue
from datetime import datetime
from utils.match import LightGlueMatcher
from utils.draw_lines import (
    draw_line_and_parallelogram,
    remove_vertical_lines,
    lightglue_detection_area,
    compare_images
)
from utils.edge import draw_edge

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
    canny_threshold2=30
    hough_rate = 44
    #line_threshold = 10
    break_rate = 35
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength= 117
    maxLineGap= 51
    parallelogram_points = None
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    background_frame = cv2.imread('/root/digit/image_backgroud.png')
    grey_base_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(grey_base_frame, (gaussian, gaussian), 0), median)
    while True:
        try:
            temp_hough = hough_rate

            frame = cv2.imread('/root/digit/image2.png')
            original_frame = cv2.imread('/root/digit/image2.png')
            height, width, channels = frame.shape
            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and help in edge detection
            blurred_image = cv2.GaussianBlur(grey_image, (gaussian, gaussian), 0)
            blurred_image = cv2.medianBlur(blurred_image,median)

            ssim_value = compare_images(blurred_base_frame,blurred_image)
            
            if ssim_value > 0.95:
                edges = np.zeros((height, width, channels), dtype=np.uint8)
                tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)
                # Place images into the layout
                tiled_layout[0:height, 0:width] = original_frame
                tiled_layout[0:height, width:width*2] = edges

                cv2.imshow("Detected Lines (in red)",tiled_layout)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                continue

            processed_frame = frame - background_frame + 128
            edges = draw_edge(processed_frame,device)

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
                        lightGlue_area = lightglue_detection_area(parallelogram_points)
                        break
            else:
                parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                lightGlue_area = lightglue_detection_area(parallelogram_points)

            tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)

            # Place images into the layout
            tiled_layout[0:height, 0:width] = original_frame
            tiled_layout[0:height, width:width*2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Detected Lines (in red)",tiled_layout)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if parallelogram_points is not None:
                    rate = count_edge_pixels_in_parallelogram(edges,parallelogram_points)
                    print(rate)
                cv2.destroyAllWindows()
                break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    process_continuous_frames()