import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from matplotlib import pyplot as plt
import math

# Load an image from file
image_path = 'screenshot/frames/*.png'
image_files = sorted(glob.glob(image_path))
rho_buffer = []
theta_buffer = []
rho_difference = []
theta_difference = []

for image_file in image_files:
    pattern1 = r'\d{6}'
    matches1 = re.findall(pattern1, image_file)
    id_string = ' '.join(matches1)
    grey_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image_file = cv2.imread(image_file)
    #print(id_string)
    # Check if the image is loaded successfully
    if grey_image is None:
        print("Error: Could not open or read the image.")
        exit()

    # Apply GaussianBlur to reduce noise and help in edge detection
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=blurred_image, threshold1=10, threshold2=0) # Canny Edge Detection

    cv2.imshow('Canny Edge Detection'+id_string, edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rate = 120
    theta_min = np.pi / 180 * 10  
    theta_max = np.pi / 180 * 170
    lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)
    for i in range(len(lines) - 1, -1, -1):
        theta = lines[i][0][1]
        if not (theta_min < theta < theta_max):
            lines = np.delete(lines, i, axis=0)  # Delete the line at index i
    last_lines = lines

    while len(lines) == 0:
        last_lines = lines
        rate -= 5
        if rate == 0:
            break
        lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)
        for i in range(len(lines) - 1, -1, -1):
            theta = lines[i][0][1]
            if not (theta_min < theta < theta_max):
                lines = np.delete(lines, i, axis=0)  # Delete the line at index i

    if not len(last_lines) == 0:
        while len(lines) > 5:
            last_lines = lines
            rate += 5
            lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)
            for i in range(len(lines) - 1, -1, -1):
                theta = lines[i][0][1]
                if not (theta_min < theta < theta_max):
                    lines = np.delete(lines, i, axis=0)  # Delete the line at index i
            if len(lines) == 0:
                lines = last_lines
                break

    sum_rho = 0
    sum_theta = 0   
    for i in range(0, len(lines)):
        sum_rho += lines[i][0][0]
        sum_theta += lines[i][0][1]

    rho = int(sum_rho/len(lines))
    theta = sum_theta/len(lines)
    
    #draw fetched line
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image_file, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imwrite('screenshot/frames/final match/final match'+id_string+'.png',image_file)
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection'+id_string, image_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
'''    if len(rho_buffer) <10:
        rho_buffer.append(rho)
        theta_buffer.append(theta)
    else:
        rho_buffer.pop(0)
        theta_buffer.pop(0)
        rho_buffer.append(rho)
        theta_buffer.append(theta)
    #print("variance of rho buffer at ",id_string,":", np.var(rho_buffer, axis=0))
    #print("average of rho buffer at ",id_string,":", np.average(rho_buffer, axis=0))
    #print("variance of theta buffer at ",id_string,":", np.var(theta_buffer, axis=0))
    #print("average of theta buffer at ",id_string,":", np.average(theta_buffer, axis=0))

    #print("difference between rho buffer and ",id_string,":", pow(rho-np.average(rho_buffer, axis=0),2))
    #print("difference between theta buffer and ",id_string,":", pow(theta-np.average(theta_buffer, axis=0),2))
    rho_difference.append(pow(rho-np.average(rho_buffer, axis=0),2))
    theta_difference.append(pow(theta-np.average(theta_buffer, axis=0),2))

fig, axs = plt.subplots(2, 1)  # 2 rows, 1 column

# First subplot
axs[0].plot(rho_difference)
axs[0].set_xlabel('Frame ID')
axs[0].set_ylabel('Rho difference')
axs[0].set_title('Plot 1')

# Second subplot
axs[1].plot(theta_difference)
axs[1].set_xlabel('Frame ID')
axs[1].set_ylabel('Theta difference')
axs[1].set_title('Plot 2')

# Adjust layout
plt.tight_layout()
plt.show()   ''' 


