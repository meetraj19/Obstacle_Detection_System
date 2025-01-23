import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the paths
model_path = '/home/thales1/Downloads/yolov8s-obb.pt'
test_images_dir = '/home/thales1/ODS4kkaggle/samples'
output_dir = '/home/thales1/ODS4kkaggle/detectionOBB'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the YOLOv8 model
try:
    model = YOLO(model_path)
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load the YOLO model: {e}")
    exit()

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

            if results is None or results.boxes is None or len(results.boxes.data) == 0:
                logging.info(f"No detections in ROI at position ({x}, {y})")
                continue  # Skip this iteration as there's nothing to process

            # Draw the bounding boxes and labels on the output image
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = model.names[int(cls)]
                cv2.rectangle(output_image, (x+x1, y+y1), (x+x2, y+y2), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x+x1, y+y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    return output_image

# Iterate through the test images
for filename in os.listdir(test_images_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(test_images_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            annotated_image = sliding_window_detection(image)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, annotated_image)
            logging.info(f"Saved annotated image: {output_path}")
        else:
            logging.warning(f"Failed to load image {filename}. It may be corrupted or in an unsupported format.")

