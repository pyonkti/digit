import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

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
            if abs(theta_degrees) < 80 or abs(theta_degrees) > 100:
                filtered_lines.append(line)
    return np.array(filtered_lines)

def score_line(line, median_angle, image_center):
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    distance_to_center = np.sqrt((x1 - image_center[0]) ** 2 + (y1 - image_center[1]) ** 2)
    
    # Score based on line length, angle similarity to the median, and distance to the center
    angle_score = 1.0 / (1.0 + abs(angle - median_angle))
    length_score = length
    center_score = 1.0 / (1.0 + distance_to_center)
    
    # Combine scores with weights (you can adjust these weights)
    return 0.5 * angle_score + 0.3 * length_score + 0.2 * center_score

def select_best_line(lines, image_shape):
    if lines is None or len(lines) == 0:
        return None
    
    # Calculate median angle
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]
    median_angle = np.median(angles)
    
    # Image center
    image_center = (image_shape[1] // 2, image_shape[0] // 2)
    
    # Score each line
    scored_lines = [(line, score_line(line, median_angle, image_center)) for line in lines]
    
    # Select the line with the highest score
    best_line = max(scored_lines, key=lambda x: x[1])[0]
    
    return best_line

def calculate_rho_theta(x1, y1, x2, y2):
    # Calculate rho and theta for the line (Hough line parameterization)
    rho = abs((x1 * y2 - y1 * x2) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    theta = math.atan2(y2 - y1, x2 - x1)
    return rho, theta

def line_to_image_edges(x1, y1, x2, y2, image):
    # Calculate the slope (m) and y-intercept (c) of the line
    if x2 != x1:  # Avoid division by zero for vertical lines
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        # Calculate points where the line crosses the left and right edges of the image
        x_start = 0
        y_start = int(c)

        x_end = image.shape[1]
        y_end = int(m * x_end + c)
    else:  # If the line is vertical
        x_start = x1
        y_start = 0
        x_end = x2
        y_end = image.shape[0]

    return (x_start, y_start), (x_end, y_end)

def draw_parallelogram_around_line(x1, y1, x2, y2, width, image):
    # Calculate the direction vector of the line
    dx = x2 - x1
    dy = y2 - y1

    # Normalize the direction vector
    length = math.sqrt(dx ** 2 + dy ** 2)
    ux = dx / length
    uy = dy / length

    # Perpendicular vector to the line
    perp_ux = -uy
    perp_uy = ux

    # Calculate the four points of the parallelogram
    offset_x = perp_ux * width
    offset_y = perp_uy * width

    pt1 = (int(x1 + offset_x), int(y1 + offset_y))
    pt2 = (int(x2 + offset_x), int(y2 + offset_y))
    pt3 = (int(x2 - offset_x), int(y2 - offset_y))
    pt4 = (int(x1 - offset_x), int(y1 - offset_y))

    # Draw the parallelogram on the image
    points = np.array([pt1, pt2, pt3, pt4], np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    return points

def lightglue_detection_area(lines,frame):
    try:
        # Find the best line segment
        best_line = select_best_line(lines, frame.shape)
        x1, y1, x2, y2 = best_line[0]

        # Calculate the points where the line crosses the image edges
        pt1, pt2 = line_to_image_edges(x1, y1, x2, y2, frame)

        # Calculate the direction vector of the line
        x1 = pt1[0] 
        y1=pt1[1]
        x2=pt2[0]
        y2=pt2[1]

        dx = x2 - x1
        dy = y2 - y1

        # Normalize the direction vector
        length = math.sqrt(dx ** 2 + dy ** 2)
        ux = dx / length
        uy = dy / length

        # Perpendicular vector to the line
        perp_ux = -uy
        perp_uy = ux

        # Calculate the four points of the parallelogram
        offset_y = perp_uy * 80

        pt1 = (int(x1), int(y1 + offset_y))
        pt2 = (int(x2), int(y2 + offset_y))
        pt3 = (int(x2), int(y2))
        pt4 = (int(x1), int(y1))

        # Draw the parallelogram on the image
        points = np.array([pt1, pt2, pt3, pt4], np.int32)
        #cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    except TypeError as e:
        print(f"An error occurred while marking detection area: {e}")
        points = None

    return points

def draw_line_and_parallelogram(lines, frame, edges, width=10):
    try:
        # Find the best line segment
        best_line = select_best_line(lines, frame.shape)
        x1, y1, x2, y2 = best_line[0]

        # Calculate the points where the line crosses the image edges
        pt1, pt2 = line_to_image_edges(x1, y1, x2, y2, frame)

        # Draw the line across the entire image
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw the parallelogram around the extended line
        #parallelogram_points = draw_parallelogram_around_line(pt1[0], pt1[1], pt2[0], pt2[1], width, frame)
        #draw_parallelogram_around_line(pt1[0], pt1[1], pt2[0], pt2[1], width, edges)

    except TypeError as e:
        print(f"An error occurred while drawing lines: {e}")
        parallelogram_points = None

    return parallelogram_points

def compare_images(imageA, imageB):
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"
    
    ssim_value, _ = ssim(imageA, imageB, full=True)

    return ssim_value



