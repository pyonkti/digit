import sys
sys.path.append('/root/digit/LightGlue')
import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

class LightGlueMatcher:
    def __init__(self):
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()
        self.matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).eval().cuda()

    def calculate_displacement(self,frame0,frame1,area0,area1):
        extracted_image0 = numpy_image_to_torch(frame0).cuda()
        extracted_image1 = numpy_image_to_torch(frame1).cuda()

        feats0 = self.extractor.extract(extracted_image0) 
        feats1 = self.extractor.extract(extracted_image1)

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        filtered_indices = filter_matches_in_region(points0, points1, area0, area1)

        if filtered_indices.size == 0:
            return None
        else:
            # If you need the filtered matches
            filtered_points0 = points0[filtered_indices]
            filtered_points1 = points1[filtered_indices]
            magnitudes = calculate_displacement(filtered_points0, filtered_points1)
            return magnitudes

def calculate_displacement(points_previous, points_current):
    displacements = points_current - points_previous
    magnitudes = np.linalg.norm(displacements, axis=1)

    return magnitudes

def extract_parallelogram_region(image, points):
    # Create a mask of the same size as the original image
    mask = np.zeros_like(image)

    # Create a filled polygon on the mask
    cv2.fillPoly(mask, [points], (255, 255, 255))  # Fill with white color

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, mask)

    return result

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def filter_matches_in_region(points0, points1, region0, region1):
    def is_point_in_region(point, region):
        x, y = point
        result = cv2.pointPolygonTest(region, (x, y), False)
        return result >= 0

    filtered_indices = []
    region_contour0 = np.array(region0, dtype=np.int32)
    region_contour1 = np.array(region1, dtype=np.int32)

    for idx, (point0, point1) in enumerate(zip(points0, points1)):
        if is_point_in_region(point0, region_contour0) and is_point_in_region(point1, region_contour1):
            filtered_indices.append(idx)

    return np.array(filtered_indices)
