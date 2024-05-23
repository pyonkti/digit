import sys
import logging
sys.path.append('/home/wei/Desktop/digit/pybind_src')
from datetime import datetime 
import cv2
import numpy as np
import optuna
from digit_interface import Digit
import grasp_object_pybind
import json


def remove_vertical_lines(lines):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        if np.pi / 180 * 10 < theta < np.pi / 180 * 170:
            filtered_lines.append(line)
    return np.array(filtered_lines)

def calculate_penalty(lines,line_threshold):
    if lines is None or len(lines) == 0:
        return 1000
    elif len(lines) <= line_threshold:
        return 0  # No penalty for 1 to 3 lines detected
    else:
        return (len(lines) - line_threshold) * 10  # Adjust the penalty factor as needed

def measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold):
    frame_count = 150  # Number of frames to analyze for stability
    rhos = []

    for _ in range(20):
        d.get_frame()

    background_frame = d.get_frame()
    blurred_base_frame = cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian_kernel, gaussian_kernel), 0)

    close_result =  grasp_object_pybind.grasp_object('172.31.1.17', 1, '0')
    if close_result != 0:
        print("error grasping")

    for _ in range(frame_count):
        frame = d.get_frame()
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey_image, (gaussian_kernel, gaussian_kernel), 0)
        blurred_image = cv2.medianBlur(blurred_image, median_kernel)
        if blurred_base_frame is not None:
            blurred_image = blurred_image - blurred_base_frame
        edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_rate)

        if lines is not None:
            lines = remove_vertical_lines(lines)
            num_lines = len(lines)
            if num_lines > 0:
                closest_line = min(lines, key=lambda line: line[0][0])
                rho, theta = closest_line[0]
                rhos.append(rho)
    

    open_result = grasp_object_pybind.open_gripper('172.31.1.17')
    if open_result != 0:
        print("error oppening")

    # Combine variance or use another metric if more appropriate
    if len(rhos) == 0:
        penalty = calculate_penalty(lines,line_threshold)
        return penalty  
    else:
        variance_rho = np.var(rhos)
        penalty = calculate_penalty(lines,line_threshold)
        return variance_rho + penalty  # Objective to minimize

def objective(trial):
    # Hyperparameters to optimize
    gaussian_kernel = trial.suggest_int('gaussian_kernel', 11, 31, step=2)  # Kernel size for Gaussian Blur must be odd
    median_kernel = trial.suggest_int('median_kernel', 3, 11, step=2)  # Kernel size for Median Blur must be odd
    canny_threshold1 = trial.suggest_int('canny_threshold1', 10, 200)
    canny_threshold2 = trial.suggest_int('canny_threshold2', 50, 300)
    hough_rate = trial.suggest_int('hough_rate', 50, 200)
    line_threshold = trial.suggest_int('line_threshold', 3, 30)

    # Create an instance of Digit, assumed already connected
    d = Digit("D20790")  # Unique serial number
    d.connect()
    variance = measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold)
    d.disconnect()

    return variance

def main():
    dt = datetime.now()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = str(dt)  # Unique identifier of the study.
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage("./"+study_name+".log"),  # NFS path for distributed optimization
    )

    study = optuna.create_study(study_name="stable grasp study", storage=storage)  # Default is to minimize
    study.optimize(objective, n_trials=50)  # Adjust number of trials based on available time and resources
    
    with open("best_trial.txt", "a") as file:
        file.write(json.dumps(study.best_trial.params))
        file.write("\n")
    print("Best parameters for stability:")
    print(study.best_trial.params)

if __name__ == "__main__":
    main()
