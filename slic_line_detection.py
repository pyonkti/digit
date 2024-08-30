import os
from matplotlib import pyplot as plt
from digit_interface import Digit
import cv2
import numpy as np
from fast_slic import Slic
from utils.draw_lines import (
    calculate_rho_theta,
    line_to_image_edges,
    draw_parallelogram_around_line,
    draw_line_and_parallelogram,
    select_best_line,
    score_line,
    remove_vertical_lines
)
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
    hough_rate = 50
    #line_threshold = 10
    break_rate = 10
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength= 50
    maxLineGap= 10
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

            edges = np.zeros((height, width), dtype=np.uint8)

            # Iterate through each pixel in the image
            for i in range(height):
                for j in range(width):
                    if assignment[i, j] == 0:
                        if ((i > 0 and assignment[i-1, j] == 1) or      # Up
                            (i < height-1 and assignment[i+1, j] == 1) or  # Down
                            (j > 0 and assignment[i, j-1] == 1) or       # Left
                            (j < width-1 and assignment[i, j+1] == 1)):   # Right
                            edges[i, j] = 255 # Mark the boundary with a value of 255

            if parallelogram_points is not None:
                rate = count_edge_pixels_in_parallelogram(edges,parallelogram_points)
                if  edge_rate_queue.__len__() < 10:
                    edge_rate_queue.enqueue(rate)
                else:
                    mean_rate = np.mean(np.array(edge_rate_queue.queue))
                    change_rate = abs((rate-mean_rate)/mean_rate)
                    if change_rate >= 0.5:
                        print('vibrated')
                        edge_rate_queue.clear()
                        parallelogram_points = None                        
                    else:
                        edge_rate_queue.enqueue(rate)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_rate, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)

            temp_hough = hough_rate

            if len(lines) == 0:
                while temp_hough > break_rate:
                    temp_hough -= threshold_increment
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
                    lines = remove_vertical_lines(lines)

                    if len(lines) > 0:
                        parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                        break
            else:
                parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)

            
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