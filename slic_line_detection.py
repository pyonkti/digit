import re
import os
import math
from matplotlib import pyplot as plt
from digit_interface import Digit
import cv2
import numpy as np
from datetime import datetime
from fast_slic import Slic
from scipy.ndimage import convolve

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

def remove_vertical_lines(lines):
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            delta_y = y2 - y1
            delta_x = x2 - x1

            # Calculate the angle in radians
            theta_radians = math.atan2(delta_y, delta_x)

            # Convert the angle to degrees
            theta_degrees = math.degrees(theta_radians)
            if np.pi / 180 * 10 < theta_degrees < np.pi / 180 * 170:
                filtered_lines.append(line)
    return np.array(filtered_lines)

def calculate_rho_theta(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    rho = abs(C) / math.sqrt(A**2 + B**2)
    theta = math.atan2(B, A)  # Correct calculation of theta
    if theta < 0:
        theta += np.pi  # Adjust theta to be in the range [0, pi]

    return rho, theta

def draw_line(lines, frame, edges):
    try:
        # Find the longest line segment (or any other criteria)
        longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
        x1, y1, x2, y2 = longest_line[0]

        # Calculate rho and theta from the endpoints
        rho, theta = calculate_rho_theta(x1, y1, x2, y2)
        a = math.cos(theta)
        b = math.sin(theta)

        length = max(frame.shape)  # Use the maximum dimension of the image
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
        pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        if b!= 0:
            parallelogram_points = np.array([
                [0, int((rho-10)/b)],
                [0, int((rho+10)/b)],
                [480, int((rho-480*a+10)/b)],
                [480, int((rho-480*a-10)/b)]
            ])
        else:
            parallelogram_points = np.array([
                [0, rho-10],
                [0, rho+10],
                [480, rho-10],
                [480, rho+10]
            ])
        draw_parallelogram(parallelogram_points, frame)
        draw_parallelogram(parallelogram_points, edges)
        # Draw the longest line
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    except TypeError as e:
        print(f"An error occurred while drawing lines: {e}")
    return parallelogram_points

def draw_parallelogram(parallelogram_points, image):
    try:
        cv2.polylines(image, [parallelogram_points], isClosed=True, color=(255, 255, 255), thickness=2)
    except TypeError as e:
        print(f"An error occurred while drawing parallelogram_: {e}")

def count_edge_pixels_in_parallelogram(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [vertices], 255)
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, mask))
    total_pixels = cv2.countNonZero(mask)
    return float(edge_pixels/total_pixels)

def process_continuous_frames(d):
    """
    Continuously captures and processes frames from the DIGIT device.
    """
    hough_rate = 44
    #line_threshold = 10
    break_rate = 10
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength= 117
    maxLineGap= 51
    parallelogram_points = None
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    edge_rate_queue = FIFOQueue(size=10)
    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()

    try:
        while True:
            temp_hough = hough_rate
            frame = d.get_frame()
            height, width, channels = frame.shape
            slic = Slic(num_components=2, compactness=10)
            assignment = slic.iterate(frame)

                # Create a copy of the image to mark the boundaries
            edges = np.zeros((height, width), dtype=np.uint8)

            # Iterate through each pixel in the image
            for i in range(height):
                for j in range(width):
                    if assignment[i, j] == 0:
                        if ((i > 0 and assignment[i-1, j] == 1) or      # Up
                            (i < height-1 and assignment[i+1, j] == 1) or  # Down
                            (j > 0 and assignment[i, j-1] == 1) or       # Left
                            (j < width-1 and assignment[i, j+1] == 1)):   # Right
                            edges[i, j] = 1 # Mark the boundary with a value of 1

            edges = edges * 255
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_rate, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)
            temp_hough = hough_rate

            if len(lines) == 0:
                while temp_hough >= break_rate:
                    temp_hough -= threshold_increment
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
                    lines = remove_vertical_lines(lines)
                    if len(lines) == 0:
                        continue
                    else:
                        parallelogram_points = draw_line(lines, frame, edges)
                        break
            else:
                parallelogram_points = draw_line(lines, frame, edges)

            
            tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)

            # Place images into the layout
            tiled_layout[0:height, 0:width] = frame
            tiled_layout[0:height, width:width*2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Detected Lines (in red)",tiled_layout)
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