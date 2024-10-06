import sys
import logging
sys.path.append('/home/wei/Desktop/digit/pybind_src')
from datetime import datetime 
import cv2
import numpy as np
import optuna
import math
from digit_interface import Digit
import grasp_object_pybind
import json

def remove_vertical_lines(lines):
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            delta_y = y2 - y1
            delta_x = x2 - x1

            # Calculate the angle in radians
            theta_radians = math.atan2(delta_y, delta_x)

            # Convert the angle to degrees
            theta_degrees = math.degrees(theta_radians)
            if np.pi / 180 * 10 < theta_degrees < np.pi / 180 * 170:
                filtered_lines.append(line)
    return np.array(filtered_lines)

def calculate_rho_theta(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    rho = abs(C) / math.sqrt(A**2 + B**2)
    theta = math.atan2(B, A)  # Correct calculation of theta
    if theta < 0:
        theta += np.pi  # Adjust theta to be in the range [0, pi]

    return rho, theta

def draw_line(lines, grey_image):
    try:
        # Find the longest line segment (or any other criteria)
        longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
        x1, y1, x2, y2 = longest_line[0]

        # Calculate rho and theta from the endpoints
        rho, theta = calculate_rho_theta(x1, y1, x2, y2)
        a = math.cos(theta)
        b = math.sin(theta)

        length = max(grey_image.shape)  # Use the maximum dimension of the image
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
        pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        # Draw the longest line
        cv2.line(grey_image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    except TypeError as e:
        print(f"An error occurred while drawing lines: {e}")

def measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold, minLineLength, maxLineGap):
    frame_count = 110  # Number of frames to analyze for stability
    rhos = []
    lines_count = []

    for _ in range(20):
        frame = d.get_frame()

    background_frame = d.get_frame()
    cv2.imshow("Detected Lines (in red)",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian_kernel, gaussian_kernel), 0), median_kernel)

    close_result =  grasp_object_pybind.grasp_object('172.31.1.17', 1, '0')
    if close_result != 0:
        print("error grasping")    

    for _ in range(frame_count):
        frame = d.get_frame()
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey_image, (gaussian_kernel, gaussian_kernel), 0)
        blurred_image = cv2.medianBlur(blurred_image, median_kernel)
        blurred_image = blurred_image - blurred_base_frame
        edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_rate, None, minLineLength, maxLineGap)
        lines = remove_vertical_lines(lines)

        if len(lines) != 0:
            longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
            x1, y1, x2, y2 = longest_line[0]

            # Calculate rho and theta from the endpoints
            rho, theta = calculate_rho_theta(x1, y1, x2, y2)
            rhos.append(rho)
            if len(lines) > line_threshold:
                lines_count.append(len(lines) - line_threshold)
            else:
                lines_count.append(0)
            draw_line(lines, frame)
        cv2.imshow("Detected Lines (in red)",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    open_result = grasp_object_pybind.open_gripper('172.31.1.17')
    if open_result != 0:
        print("error oppening")

    # Combine variance or use another metric if more appropriate
    if len(rhos) > 0:
        rho_std = np.std(rhos)
        print(f"consequtiveness: {1000*(1-len(rhos)/frame_count)}")
        print(f"standard variation of rho: {rho_std}")
        print(f"number of average exceeded lines: {sum(lines_count)/len(lines_count)}")
        return 1000*(1-len(rhos)/frame_count) + rho_std + sum(lines_count)/len(lines_count)  # Objective to minimize
    else:
        return 1000

def objective(trial):
    # Hyperparameters to optimize
    gaussian_kernel = trial.suggest_int('gaussian_kernel', 13, 23, step=2)  # Kernel size for Gaussian Blur must be odd
    median_kernel = trial.suggest_int('median_kernel', 3, 9, step=2)  # Kernel size for Median Blur must be odd
    canny_threshold1 = trial.suggest_int('canny_threshold1', 10, 200)
    canny_threshold2 = trial.suggest_int('canny_threshold2', 50, 300)
    hough_rate = trial.suggest_int('hough_rate', 30, 200)
    line_threshold = trial.suggest_int('line_threshold', 3, 25)
    minLineLength= trial.suggest_int('minLineLength', 10, 300)
    maxLineGap= trial.suggest_int('maxLineGap', 0, 100)

    # Create an instance of Digit, assumed already connected
    d = Digit("D20782")  # Unique serial number
    d.connect()
    variance = measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold, minLineLength, maxLineGap)
    d.disconnect()

    return variance

def main():
    storage = "sqlite:///optuna.db"

    study = optuna.create_study(study_name="stable grasp study", storage=storage, load_if_exists = True, direction = 'minimize')  # Default is to minimize
    study.optimize(objective, n_trials=30)  # Adjust number of trials based on available time and resources
    
    with open("best_trial.txt", "a") as file:
        file.write(json.dumps(study.best_trial.params))
        file.write("\n")
    print("Best parameters for stability:")
    print(study.best_trial.params)

if __name__ == "__main__":
    main()
