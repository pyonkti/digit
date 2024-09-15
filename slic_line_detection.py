import os
from digit_interface import Digit
import cv2
import numpy as np
from datetime import datetime
from fast_slic import Slic
from utils.match import LightGlueMatcher
from utils.draw_lines import (
    draw_line_and_parallelogram,
    remove_vertical_lines,
    lightglue_detection_area,
    compare_images
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
    gaussian = 23
    median = 5
    hough_rate = 50
    break_rate = 10
    threshold_increment = 1
    minLineLength = 117
    maxLineGap = 51
    parallelogram_points = None
    match_counter = 10
    lightGlue_area = None
    matchFrame = None
    detach_flag = False
    detach_counter = 30
    matcher = LightGlueMatcher()
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True) 
    edge_rate_queue = FIFOQueue(size=10)

    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()

    background_frame = d.get_frame()
    blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian, gaussian), 0), median)

    try:
        while True:
            temp_hough = hough_rate
            former_parallelogram_points = parallelogram_points
            frame = d.get_frame()
            height, width, channels = frame.shape

            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grey_image, (gaussian, gaussian), 0)
            blurred_image = cv2.medianBlur(blurred_image,median)

            if blurred_base_frame is not None:  
                ssim_value = compare_images(blurred_base_frame,blurred_image)

            if former_parallelogram_points is None and ssim_value > 0.92:
                if detach_flag and detach_counter>0:
                    detach_counter -= 1
                elif detach_flag:
                    detach_flag = False
                    detach_counter = 30
                    print('Component detached')

                edges = np.zeros((height, width, channels), dtype=np.uint8)
                
                tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)
                tiled_layout[0:height, 0:width] = frame
                tiled_layout[0:height, width:width*2] = edges

                cv2.imshow("Detected Lines (in red)",tiled_layout)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            slic = Slic(num_components=2, compactness=10)
            assignment = slic.iterate(frame)

            edges = np.zeros((height, width), dtype=np.uint8)

            gradient_x = np.abs(np.diff(assignment, axis=1))
            gradient_y = np.abs(np.diff(assignment, axis=0))

            gradient_x = np.pad(gradient_x, ((0, 0), (0, 1)), mode='constant', constant_values=0)
            gradient_y = np.pad(gradient_y, ((0, 1), (0, 0)), mode='constant', constant_values=0)

            edges = np.clip(gradient_x + gradient_y, 0, 1) * 255
            edges = edges.astype(np.uint8)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_rate, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)

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

            if parallelogram_points is not None:
                if detach_flag:
                    detach_flag = False
                    detach_counter = 30
                match_counter -= 1
                if match_counter == 0 and matchFrame is None:
                    match_counter = 10
                    lightGlue_area = lightglue_detection_area(lines,frame)
                    matchFrame = frame
                elif match_counter == 0:
                    match_counter = 10
                    new_lightGlue_area = lightglue_detection_area(lines,frame)

                    magnitudes = matcher.calculate_displacement(matchFrame,frame,lightGlue_area,new_lightGlue_area)

                    lightGlue_area = new_lightGlue_area
                    matchFrame = frame

                    mean_magnitude = np.mean(magnitudes)
                    if mean_magnitude > 10:
                        print("Componet attached gentlely")
            
            if former_parallelogram_points is not None:
                rate = count_edge_pixels_in_parallelogram(edges,former_parallelogram_points)
                if  edge_rate_queue.__len__() < 10 and rate >= 0.02:
                    edge_rate_queue.enqueue(rate)
                elif edge_rate_queue.__len__() < 10 and rate < 0.02:
                    edge_rate_queue.clear()
                elif parallelogram_points is not None:
                    mean_rate = np.mean(np.array(edge_rate_queue.queue))
                    change_rate = abs((rate-mean_rate)/mean_rate)
                    if change_rate >= 0.5:
                        print('Component jumped')
                        edge_rate_queue.clear()
                    else:
                        edge_rate_queue.enqueue(rate)
                else:
                    detach_flag = True
                    print('Component disappeared')
            
            tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)
            tiled_layout[0:height, 0:width] = frame
            tiled_layout[0:height, width:width*2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Detected Lines (in red)",tiled_layout)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    d = Digit("D20812") # Unique serial number
    d.connect()
    process_continuous_frames(d)
    d.disconnect()