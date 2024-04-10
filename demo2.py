import re
import math
from matplotlib import pyplot as plt
from digit_interface import Digit
from digit_interface import DigitHandler
import cv2
import numpy as np

def remove_vertical_lines(lines):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        if np.pi / 180 * 10 < theta < np.pi / 180 * 170:
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
        
def process_continuous_frames(d,base_frame):
    """
    Continuously captures and processes frames from the DIGIT device.
    """
    try:
        while True:
            frame = d.get_frame()

            if not isinstance(frame, np.ndarray):
                print("Error: Frame is not a valid numpy array.")
                exit()

            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and help in edge detection
            blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
            blurred_image = blurred_image - base_frame

            # Canny Edge Detection
            edges = cv2.Canny(image=blurred_image, threshold1=0, threshold2=10) # Canny Edge Detection

            rate = 160
            break_rate = 60
            lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)

            if lines is None:
                cv2.imshow("Detected Lines (in red)", grey_image)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                lines = remove_vertical_lines(lines)            
                last_lines = lines
                while len(lines) == 0:
                    last_lines = lines
                    rate -= 5
                    if rate <= break_rate:
                        break
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)
                    lines = remove_vertical_lines(lines)
                    if len(lines) != 0:
                        print('should be a line 1')
                        draw_line(lines, grey_image)
                        break

                if len(last_lines) > 0:
                    if 1 <= len(lines) <= 5:
                        draw_line(lines, grey_image)
                    while len(lines) > 5:
                        last_lines = lines
                        rate += 5
                        lines = cv2.HoughLines(edges, 1, np.pi / 180, rate, None, 0, 0)
                        lines = remove_vertical_lines(lines)
                        print('should be a line 2')
                        if len(lines) == 0:
                            lines = last_lines 
                            draw_line(lines, grey_image)
                            break   

            cv2.imshow("Detected Lines (in red)", grey_image)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


d = Digit("D20782") # Unique serial number
d.connect()
base_frame = d.get_frame()
grey_image = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)

process_continuous_frames(d,blurred_image)
d.disconnect()