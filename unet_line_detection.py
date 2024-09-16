import os
from digit_interface import Digit
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from edge_detection_autoencoder import UNet
from utils.match import LightGlueMatcher
from utils.draw_lines import (
    draw_line_and_parallelogram,
    remove_vertical_lines,
    lightglue_detection_area,
    compare_images
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

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

def count_edge_pixels_in_parallelogram(output_resized, vertices):
    mask = np.zeros_like(output_resized)
    cv2.fillPoly(mask, [vertices], 255)
    edge_pixels = cv2.countNonZero(cv2.bitwise_and(output_resized, mask))
    total_pixels = cv2.countNonZero(mask)
    return float(edge_pixels/total_pixels)

def process_continuous_frames(d):
    """
    Continuously captures and processes frames from the DIGIT device.
    """

    lines_flag = False
    model_path = '/home/wei/Desktop/digit/digit/edge_detection_autoencoder/unet_line_detection.pth'
    gaussian = 23
    median = 5
    hough_rate = 44
    break_rate = 35
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength = 117
    maxLineGap = 51
    match_counter = 3
    lightGlue_area = None
    matchFrame = None
    detach_flag = False
    detach_counter = 30
    matcher = LightGlueMatcher()
    parallelogram_points = None
    former_parallelogram_points = None
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    edge_rate_queue = FIFOQueue(size=10)

    file_path = '/home/wei/Desktop/digit/digit/outcome_log/unet_log.txt'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(file_path, 'a') as file:
        file.write( f'Timestamp: {timestamp}' + '\n')

    model = UNet(in_channels=3, out_channels=1).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()

    background_frame = d.get_frame()
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
            
            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grey_image, (gaussian, gaussian), 0)
            blurred_image = cv2.medianBlur(blurred_image,median)

            if blurred_base_frame is not None:  
                ssim_value = compare_images(blurred_base_frame,blurred_image)

            if former_parallelogram_points is None and ssim_value > 0.9:
                if detach_flag and detach_counter>0:
                    detach_counter -= 1
                elif detach_flag:
                    detach_flag = False
                    detach_counter = 30
                    print('Component detached')
                    with open(file_path, 'a') as file:
                            file.write('Component detached' + '\n')
                edges = np.zeros((height, width, channels), dtype=np.uint8)
                tiled_layout = np.zeros((height, width * 3, channels), dtype=np.uint8)
                tiled_layout[0:height, 0:width] = frame
                tiled_layout[0:height, width:width*2] = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
                tiled_layout[0:height, width*2:width*3] = edges

                cv2.imshow("Detected Lines (in red)",tiled_layout)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            original_image = Image.fromarray(frame).convert("RGB")

            image_tensor = transform(original_image).unsqueeze(0).cuda()

            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                output = torch.sigmoid(output)
                output = output.squeeze().cpu()
            
            output_resized = F.resize(output.unsqueeze(0), [height, width]).squeeze(0)
            output_resized = output_resized.numpy()
            edges = (output_resized * 255).astype(np.uint8)

            # First attempt to find lines with initial rate
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)

            # Adjust rate until lines are found, adhering to break_rate limit
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
                    lightGlue_area = lightglue_detection_area(lines,frame)
                    matchFrame = frame
                elif match_counter == 0:
                    match_counter = 10
                    new_lightGlue_area = lightglue_detection_area(lines,frame)

                    magnitudes = matcher.calculate_displacement(matchFrame,frame,lightGlue_area,new_lightGlue_area)

                    lightGlue_area = new_lightGlue_area
                    matchFrame = frame

                    if magnitudes is not None and len(magnitudes) > 0:
                        mean_magnitude = np.mean(magnitudes)
                    else:
                        mean_magnitude = 0

                    print(mean_magnitude)

                    if mean_magnitude > 14.5 and mean_magnitude < 25:
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
                        print('Component jumped')
                        with open(file_path, 'a') as file:
                            file.write('Component jumped' + '\n')
                        edge_rate_queue.clear()
                        detach_flag = True
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