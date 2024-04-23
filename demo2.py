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

def draw_line(lines, grey_image):
    if lines is None or len(lines) == 0:
        print("No lines found")
        return  # No lines to process

    try:
        # Find the line with the smallest rho value
        closest_line = min(lines, key=lambda line: line[0][0])
        #rho = sum(line[0][0] for line in lines) / len(lines)
        #theta = sum(line[0][1] for line in lines) / len(lines)
        rho, theta = closest_line[0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        # Draw the closest line
        cv2.line(grey_image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    except TypeError as e:
        print(f"An error occurred: {e}")
        
def process_continuous_frames(d):
    """
    Continuously captures and processes frames from the DIGIT device.
    """
    frame_id = 0
    try:
        while True:
            frame = d.get_frame()
            if not isinstance(frame, np.ndarray):
                print("Error: Frame is not a valid numpy array.")
                exit()
            if frame_id < 10:  
                frame_id += 1
                continue
            if frame_id == 10:
                blurred_base_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                frame_id += 1

            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and help in edge detection
            blurred_image = cv2.GaussianBlur(grey_image, (7, 7), 0)
            blurred_image = cv2.medianBlur(blurred_image,7)

            if blurred_base_frame is not None:
                blurred_image = blurred_image - blurred_base_frame

            # Canny Edge Detection
            edges = cv2.Canny(image=blurred_image, threshold1=20, threshold2=10) # Canny Edge Detection

            rate = 100
            break_rate = 70
            threshold_increment = 5  # How much to change the threshold by in each iteration
            found_lines = False

            # First attempt to find lines with initial rate
            lines = cv2.HoughLines(edges, 1, np.pi / 180, rate)

            # Adjust rate until lines are found, adhering to break_rate limit
            while lines is None or len(remove_vertical_lines(lines)) == 0:
                if rate >= break_rate:
                    rate -= threshold_increment
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, rate)
                else:
                    break

            # If lines are found, filter out vertical lines
            if lines is not None:
                lines = remove_vertical_lines(lines)
                if len(lines) > 0:
                    found_lines = True

            # If more than 3 lines are found, try to narrow it down by increasing the threshold
            if found_lines and len(lines) > 8:
                while len(lines) > 8: 
                    rate += threshold_increment
                    temp_lines = cv2.HoughLines(edges, 1, np.pi / 180, rate)
                    if temp_lines is None:
                        draw_line(lines, frame)
                        break
                    elif temp_lines is not None:
                        temp_lines = remove_vertical_lines(temp_lines)
                        if 0 < len(temp_lines) <= 8:
                            lines = temp_lines  # Update lines with the filtered results
                            draw_line(lines, frame)
                        else:
                            break  # Exit the loop if no lines are found in the current iteration
            elif found_lines and 0 < len(lines) <= 8:
                draw_line(lines, frame)

            cv2.imshow("Detected Lines (in red)", frame)
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