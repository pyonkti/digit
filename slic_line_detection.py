import os
from digit_interface import (
    Digit
)
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
    hough_rate = 44
    break_rate = 35
    threshold_increment = 1
    minLineLength = 117
    maxLineGap = 51
    parallelogram_points = None
    match_counter = 1
    lightGlue_area = None
    matchFrame = None
    detach_flag = False
    detach_counter = 30
    matcher = LightGlueMatcher()
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True) 
    edge_rate_queue = FIFOQueue(size=10)
    former_parallelogram_points = None
    lines_flag = False

    file_path = '/home/wei/Desktop/digit/digit/outcome_log/slic_log.txt'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(file_path, 'a') as file:
        file.write( f'Timestamp: {timestamp}' + '\n')


    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()

    background_frame = d.get_frame()
    background_frame = background_frame[:, :, 0]
    background_frame = np.stack((background_frame,)*3, axis=-1)  # Convert to 3 channels
    blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian, gaussian), 0), median)

    try:
        while True:
            temp_hough = hough_rate
            if lines_flag:
                former_parallelogram_points = parallelogram_points
            else:
                former_parallelogram_points = None
            lines_flag = False

            frame = d.get_frame()
            height, width, channels = frame.shape
            original_frame = frame
            frame = frame[:, :, 0]
            frame = np.stack((frame,)*3, axis=-1)  # Convert to 3 channels

            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grey_image, (gaussian, gaussian), 0)
            blurred_image = cv2.medianBlur(blurred_image,median)

            if blurred_base_frame is not None:  
                ssim_value = compare_images(blurred_base_frame,blurred_image)
            
            if ssim_value > 0.9:
                if detach_flag:
                    print(detach_counter)
                    if detach_counter > 0:
                        detach_counter -= 1
                    else:
                        detach_flag = False
                        detach_counter = 30
                        print('Component detached')
                        with open(file_path, 'a') as file:
                            file.write('Component detached' + '\n')

                edges = np.zeros((height, width, channels), dtype=np.uint8)
                tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)
                tiled_layout[0:height, 0:width] = frame
                tiled_layout[0:height, width:width*2] = edges

                cv2.imshow("Detected Lines (in red)",tiled_layout)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            slic = Slic(num_components=2, compactness=1)
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

            if lines is None or len(lines) == 0:
                while temp_hough > break_rate:
                    temp_hough -= threshold_increment
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
                    lines = remove_vertical_lines(lines)

                    if lines is not None and len(lines) > 0:
                        parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                        lines_flag = True
                        break
            else:
                parallelogram_points = draw_line_and_parallelogram(lines, frame, edges, width=10)
                lines_flag = True

            if lines_flag:
                match_counter -= 1
                if detach_flag:
                    detach_flag = False
                    detach_counter = 30
                if match_counter == 0 and matchFrame is None:
                    match_counter = 10
                    lightGlue_area = lightglue_detection_area(lines,original_frame)
                    matchFrame = frame
                elif match_counter == 0:
                    match_counter = 10
                    new_lightGlue_area = lightglue_detection_area(lines,original_frame)

                    magnitudes = matcher.calculate_displacement(matchFrame,frame,lightGlue_area,new_lightGlue_area)

                    lightGlue_area = new_lightGlue_area
                    matchFrame = frame

                    if magnitudes is not None and len(magnitudes) > 0:
                        mean_magnitude = np.mean(magnitudes)
                    else:
                        mean_magnitude = 0

                    #print(mean_magnitude)
                    
                    if mean_magnitude > 14.5 and mean_magnitude <25:
                        print('Componet attached gently')
                        with open(file_path, 'a') as file:
                            file.write('Componet attached gently' + '\n')
            
            if former_parallelogram_points is not None:
                rate = count_edge_pixels_in_parallelogram(edges,former_parallelogram_points)
                if  edge_rate_queue.__len__() < 10 and rate >= 0.02:
                    edge_rate_queue.enqueue(rate)
                elif edge_rate_queue.__len__() < 10 and rate < 0.02:
                    edge_rate_queue.clear()
                elif lines_flag:
                    mean_rate = np.mean(np.array(edge_rate_queue.queue))
                    change_rate = abs((rate-mean_rate)/mean_rate)
                    if change_rate >= 0.5:
                        detach_flag = True
                        print('Component jumped')
                        with open(file_path, 'a') as file:
                            file.write('Component jumped' + '\n')
                        edge_rate_queue.clear()
                    else:
                        edge_rate_queue.enqueue(rate)
                else:
                    detach_flag = True
                    print('Component disappeared')
                    with open(file_path, 'a') as file:
                        file.write('Component disappeared' + '\n')
            
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
    d.set_intensity_rgb(intensity_r = 0, intensity_g = 0, intensity_b = 15)
    process_continuous_frames(d)
    d.disconnect()