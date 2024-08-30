import os
import math
from matplotlib import pyplot as plt
from digit_interface import Digit
import cv2
import numpy as np
import torch
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from edge_detection_autoencoder import UNet
from utils.draw_lines import (
    calculate_rho_theta,
    line_to_image_edges,
    draw_parallelogram_around_line,
    draw_line_and_parallelogram,
    select_best_line,
    score_line,
    remove_vertical_lines
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
    model_path = '/home/wei/Desktop/digit/digit/edge_detection_autoencoder/unet_line_detection.pth'
    hough_rate = 44
    break_rate = 35
    threshold_increment = 1  # How much to change the threshold by in each iteration
    minLineLength= 117
    maxLineGap= 51
    parallelogram_points = None
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    edge_rate_queue = FIFOQueue(size=10)

    model = UNet(in_channels=3, out_channels=1).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #skip first 20 frames for camera to adjust its white balance
    for _ in range(20):
        d.get_frame()

    try:
        while True:
            temp_hough = hough_rate

            frame = d.get_frame()
            height, width, channels = frame.shape
            
            original_image = Image.fromarray(frame).convert("RGB")
            original_size = original_image.size

            image_tensor = transform(original_image).unsqueeze(0).cuda()

            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                output = torch.sigmoid(output)
                output = output.squeeze().cpu()
            
            output_resized = F.resize(output.unsqueeze(0), [original_size[1], original_size[0]]).squeeze(0)
            output_resized = output_resized.numpy()
            output_resized = (output_resized * 255).astype(np.uint8)
            original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            
            if parallelogram_points is not None:
                rate = count_edge_pixels_in_parallelogram(output_resized,parallelogram_points)
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

            # First attempt to find lines with initial rate
            lines = cv2.HoughLinesP(output_resized, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
            lines = remove_vertical_lines(lines)

            # Adjust rate until lines are found, adhering to break_rate limit
            if len(lines) == 0:
                while temp_hough > break_rate:
                    temp_hough -= threshold_increment
                    lines = cv2.HoughLinesP(output_resized, 1, np.pi / 180, temp_hough, None, minLineLength, maxLineGap)
                    lines = remove_vertical_lines(lines)

                    if len(lines) > 0:
                        parallelogram_points = draw_line_and_parallelogram(lines, original_image_bgr, output_resized, width=10)
                        break
            else:
                parallelogram_points = draw_line_and_parallelogram(lines, original_image_bgr, output_resized, width=10)

            tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)

            original_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
            # Place images into the layout
            tiled_layout[0:height, 0:width] = original_image
            tiled_layout[0:height, width:width*2] = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)

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