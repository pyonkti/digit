import cv2
import numpy as np
import optuna
from digit_interface import Digit

def remove_vertical_lines(lines):
    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        if np.pi / 180 * 10 < theta < np.pi / 180 * 170:
            filtered_lines.append(line)
    return np.array(filtered_lines)

def measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold):
    frame_count = 150  # Number of frames to analyze for stability
    rhos = []
    thetas = []
    total_lines_detected = 0

    for _ in range(20):
        d.get_frame()

    background_frame = d.get_frame()
    blurred_base_frame = cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (gaussian_kernel, gaussian_kernel), 0)

    cv2.imshow("Position the object and press any key to continue", background_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for _ in range(frame_count):
        frame = d.get_frame()
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey_image, (gaussian_kernel, gaussian_kernel), 0)
        blurred_image = cv2.medianBlur(blurred_image, median_kernel)
        if blurred_base_frame is not None:
            blurred_image = blurred_image - blurred_base_frame
        edges = cv2.Canny(blurred_image, canny_threshold1, canny_threshold2)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_rate)
        lines = remove_vertical_lines(lines) if lines is not None else []

        if lines is not None:
            closest_line = min(lines, key=lambda line: line[0][0])
            rho, theta = closest_line[0]
            rhos.append(rho)
            total_lines_detected += len(lines)

    variance_rho = np.var(rhos)
    penalty = 1000 * (frame_count - total_lines_detected) if total_lines_detected < 5 else 0
 
    cv2.imshow("Detach the object and press any key to continue", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Combine variance or use another metric if more appropriate
    return variance_rho + penalty  # Objective to minimize

def objective(trial):
    # Hyperparameters to optimize
    gaussian_kernel = trial.suggest_int('gaussian_kernel', 3, 15, step=2)  # Kernel size for Gaussian Blur must be odd
    median_kernel = trial.suggest_int('median_kernel', 3, 11, step=2)  # Kernel size for Median Blur must be odd
    canny_threshold1 = trial.suggest_int('canny_threshold1', 10, 200)
    canny_threshold2 = trial.suggest_int('canny_threshold2', 50, 300)
    hough_rate = trial.suggest_int('hough_rate', 50, 200)
    line_threshold = trial.suggest_int('line_threshold', 3, 20)

    # Create an instance of Digit, assumed already connected
    d = Digit("D20790")  # Unique serial number
    d.connect()
    variance = measure_stability(d, gaussian_kernel, median_kernel, canny_threshold1, canny_threshold2, hough_rate, line_threshold)
    d.disconnect()

    return variance

def main():
    study = optuna.create_study()  # Default is to minimize
    study.optimize(objective, n_trials=50)  # Adjust number of trials based on available time and resources

    print("Best parameters for stability:")
    print(study.best_trial.params)

if __name__ == "__main__":
    main()
