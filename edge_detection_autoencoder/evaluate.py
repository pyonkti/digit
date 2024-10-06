import torch
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from unet import UNet

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def calculate_rho_theta(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    rho = abs(C) / math.sqrt(A**2 + B**2)
    theta = math.atan2(B, A)  # Correct calculation of theta
    if theta < 0:
        theta += np.pi  # Adjust theta to be in the range [0, pi]

    return rho, theta

def draw_line(lines, frame):
    try:
        # Find the longest line segment (or any other criteria)
        longest_line = max(lines, key=lambda line: np.linalg.norm((line[0][2] - line[0][0], line[0][3] - line[0][1])))
        x1, y1, x2, y2 = longest_line[0]

        # Calculate rho and theta from the endpoints
        rho, theta = calculate_rho_theta(x1, y1, x2, y2)
        a = math.cos(theta)
        b = math.sin(theta)

        length = max(frame.shape)  # Use the maximum dimension of the image
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
        pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        if b!= 0:
            parallelogram_points = np.array([
                [0, int((rho-10)/b)],
                [0, int((rho+10)/b)],
                [480, int((rho-480*a+10)/b)],
                [480, int((rho-480*a-10)/b)]
            ])
        else:
            parallelogram_points = np.array([
                [0, rho-10],
                [0, rho+10],
                [480, rho-10],
                [480, rho+10]
            ])
        draw_parallelogram(parallelogram_points, frame)
        # Draw the longest line
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    except TypeError as e:
        print(f"An error occurred while drawing lines: {e}")
    return parallelogram_points

def draw_parallelogram(parallelogram_points, image):
    try:
        cv2.polylines(image, [parallelogram_points], isClosed=True, color=(255, 255, 255), thickness=2)
    except TypeError as e:
        print(f"An error occurred while drawing parallelogram_: {e}")

def evaluate(model_path, image_path):
    # Load the model
    model = UNet(in_channels=3, out_channels=1).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load the original image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    
    # Transform the image for model input
    image_tensor = transform(original_image).unsqueeze(0).cuda()
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu()
    
    # Resize the output mask to the original image size

    threshold = 0.3  # Threshold value, typically 0.5 for binary classification
    output_resized = F.resize(output.unsqueeze(0), [original_size[1], original_size[0]]).squeeze(0)
    output_resized = (output_resized > threshold).float()
    output_resized = output_resized.numpy()
    output_resized = (output_resized * 255).astype(np.uint8)
    original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    lines = cv2.HoughLinesP(output_resized, 1, np.pi / 180, 44, None, 117, 51)
    parallelogram_points = draw_line(lines, original_image_bgr)

    original_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

    # Display the original image and the resized output mask
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title('Detected Lines')
    plt.imshow(output_resized, cmap='gray')
    plt.show()

# Example usage
evaluate('unet_line_detection(2).pth', '/root/digit/edge_detection_autoencoder/dataset/images/detected_lines_20240806_173354.png')
