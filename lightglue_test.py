import sys
sys.path.append('/root/digit/LightGlue')
import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from utils.draw_lines import (
    remove_vertical_lines,
    lightglue_detection_area
)

class CannyEdgeDetector:
    def __init__(self):
        self.gaussian = 23
        self.median = 5
        self.canny_threshold1=20
        self.canny_threshold2=60
        self.hough_rate = 44
        self.break_rate = 35
        self.threshold_increment = 1 
        self.minLineLength= 117
        self.maxLineGap= 51
        self.background_frame = cv2.imread('/root/digit/edge_detection_autoencoder/dataset/empty/detected_lines_20240820_141634.png')
        self.blurred_base_frame = cv2.medianBlur(cv2.GaussianBlur(cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2GRAY), (self.gaussian, self.gaussian), 0), self.median)

    def locate_edge(self,frame):
        lightGlue_area = None

        # Apply GaussianBlur to reduce noise and help in edge detection
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grey_image, (self.gaussian, self.gaussian), 0)
        blurred_image = cv2.medianBlur(blurred_image,self.median)

        blurred_image = cv2.absdiff(blurred_image, self.blurred_base_frame)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(blurred_image)
        # Canny Edge Detection
        edges = cv2.Canny(image=enhanced_image, threshold1=self.canny_threshold1, threshold2=self.canny_threshold2) # Canny Edge Detection
        
        temp_hough = self.hough_rate        
        # First attempt to find lines with initial rate
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, self.minLineLength, self.maxLineGap)
        lines = remove_vertical_lines(lines)
        
        # Adjust rate until lines are found, adhering to break_rate limit
        if len(lines) == 0:
            while temp_hough > self.break_rate:
                temp_hough -= self.threshold_increment
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, temp_hough, None, self.minLineLength, self.maxLineGap)
                lines = remove_vertical_lines(lines)

                if len(lines) > 0:
                    lightGlue_area = lightglue_detection_area(lines,frame)
                    break
        else:
            lightGlue_area = lightglue_detection_area(lines,frame)
        return lightGlue_area

def extract_parallelogram_region(image, points):
    # Create a mask of the same size as the original image
    mask = np.zeros_like(image)

    # Create a filled polygon on the mask
    cv2.fillPoly(mask, [points], (255, 255, 255))  # Fill with white color

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, mask)

    return result

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def draw_matches(image0, image1, points0, points1):

    # Convert images to BGR format for OpenCV
    image0_np = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1_np = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

    # Create a combined image to draw matches
    combined_image = np.concatenate((image0_np, image1_np), axis=1)

    # Offset to shift points on the second image
    offset = image0_np.shape[1]

    # Draw keypoints and matches
    for pt0, pt1 in zip(points0, points1):
        # Convert points to int
        pt0 = tuple(map(int, pt0))
        pt1 = tuple(map(int, pt1))

        # Draw keypoints on the first image
        cv2.circle(combined_image, pt0, 5, (0, 0, 255), -1)

        # Draw keypoints on the second image (shift points to the right)
        pt1_shifted = (pt1[0] + offset, pt1[1])
        cv2.circle(combined_image, pt1_shifted, 5, (0, 0, 255), -1)

        # Draw line connecting the keypoints
        cv2.line(combined_image, pt0, pt1_shifted, (0, 0, 255), 1)

    # Show the combined image
    cv2.imshow('Matches', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_displacement(points_previous, points_current):
    # Calculate the displacement vectors
    displacements = points_current - points_previous

    # Calculate the magnitudes of the displacement vectors
    magnitudes = np.linalg.norm(displacements, axis=1)

    return displacements, magnitudes

if __name__ =='__main__':
    edgeDetector  = CannyEdgeDetector()
    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).eval().cuda()  # load the matcher

    image0 = cv2.imread('/root/digit/screenshot/consecutive1/2024-03-08 16-49-15180.png')
    image1 = cv2.imread('/root/digit/screenshot/consecutive1/2024-03-08 16-49-15190.png')
    
    area0 = edgeDetector.locate_edge(image0)
    area1 = edgeDetector.locate_edge(image1)

    if area0 is not None and area1 is not None:
        extracted_region0 = extract_parallelogram_region(image0, area0)
        extracted_region1 = extract_parallelogram_region(image1, area1)
        extracted_image0 = numpy_image_to_torch(extracted_region0).cuda()
        extracted_image1 = numpy_image_to_torch(extracted_region1).cuda()
    else:
        extracted_image0 = numpy_image_to_torch(image0).cuda()
        extracted_image1 = numpy_image_to_torch(image1).cuda()

    # extract local features
    feats0 = extractor.extract(extracted_image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(extracted_image1)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    points0 = points0.cpu().numpy()
    points1 = points1.cpu().numpy()

    displacements, magnitudes = calculate_displacement(points0, points1)
    print("Displacements:", displacements)
    print("Magnitudes:", magnitudes)
    draw_matches(image0, image1, points0, points1)