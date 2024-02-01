import cv2
import numpy as np
import matplotlib.pyplot as plt
import re


# Load an image from file
image_path = 'digit-interface/screenshot/parallel sensors/Digit View D20790_screenshot2_01.02.2024.png'
pattern1 = r'D\d{5}'
pattern2 = r'screenshot\d{0,1}'
matches1 = re.findall(pattern1, image_path)
matches2 = re.findall(pattern2, image_path)
id_string = ' '.join(matches1)
number_string =' '.join(matches2[1])
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not open or read the image.")
    exit()

# Apply GaussianBlur to reduce noise and help in edge detection
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Canny Edge Detection
edges = cv2.Canny(image=blurred_image, threshold1=30, threshold2=30) # Canny Edge Detection

cv2.imwrite('digit-interface/screenshot/parallel sensors/edge detection/Canny Edge Detection'+id_string+'_'+number_string+'.jpg',edges)
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()

