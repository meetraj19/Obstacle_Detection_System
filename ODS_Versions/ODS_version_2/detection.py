import os
import cv2
import numpy as np
from ultralytics import YOLO

# Set the paths
model_path = '/home/thales1/ODSv02/pythonProject/odsv02/runs/detect/train11/weights/best.pt'
test_images_dir = '/datasetsample/test/images'
output_dir = '/datasetsample/test_results'

# Load the YOLOv8 model
model = YOLO(model_path)

# Set the sliding window parameters
window_size = (320, 240)
step_size = (160, 120)

# Function to perform sliding window object detection
def sliding_window_detection(image):
    output_image = image.copy()

    # Perform the sliding window approach
    for x in range(0, image.shape[1] - window_size[0] + 1, step_size[0]):
        for y in range(0, image.shape[0] - window_size[1] + 1, step_size[1]):
            # Extract the ROI
            roi = image[y:y+window_size[1], x:x+window_size[0]]

            # Run the YOLOv8 model on the ROI
            results = model.predict(roi)[0]

            # Draw the bounding boxes and labels on the output image
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[int(cls)]
                cv2.rectangle(output_image, (x+x1, y+y1), (x+x2, y+y2), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x+x1, y+y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    return output_image

# Iterate through the test images
for filename in os.listdir(test_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(test_images_dir, filename)
        image = cv2.imread(image_path)
