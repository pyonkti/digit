import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from digit_interface import Digit
from digit_interface import DigitHandler
import math

def remove_vertical_lines(lines):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        if (np.pi / 180) * 10 < theta < (np.pi / 180) * 170:
            filtered_lines.append(line)
    return np.array(filtered_lines)

def draw_line(lines,grey_image):
    sum_rho = 0
    sum_theta = 0   
    for i in range(0, len(lines)):
        sum_rho += lines[i][0][0]
        sum_theta += lines[i][0][1]

    rho = int(sum_rho/len(lines))
    theta = sum_theta/len(lines)
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(grey_image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

d = Digit("D20782") # Unique serial number
d.connect()

# Load an image from file
image_path = 'screenshot/frames/2024-03-08 16-48-410001.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
'''if image is None:
    print("Error: Could not open or read the image.")
    exit()'''

frame = d.get_frame()
# Apply GaussianBlur to reduce noise and help in edge detection
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Canny Edge Detection
edges = cv2.Canny(image=blurred_image, threshold1=30, threshold2=30) # Canny Edge Detection
rate = 50
lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)

if lines is None:
    cv2.imshow('Canny Edge Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    lines = remove_vertical_lines(lines)
    draw_line(lines, image)
    cv2.imshow('Canny Edge Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv2.imwrite('digit-interface/screenshot/parallel sensors/edge detection/Canny Edge Detection'+id_string+'_'+number_string+'.jpg',edges)
# Display Canny Edge Detection Image

