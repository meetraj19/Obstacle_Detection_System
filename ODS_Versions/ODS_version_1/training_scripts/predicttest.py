import os
import cv2
import numpy as np
from ultralytics import YOLO

# Set the paths
model_path = '/home/thales1/ODSv02/pythonProject/runs/detect/train20/weights/last.pt'
test_images_dir = '/datasetsample/test/images'
output_dir = '/datasetsample/test_results'

# Load the YOLOv8 model
model = YOLO(model_path)

# Iterate through the test images
for filename in os.listdir(test_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(test_images_dir, filename)
        image = cv2.imread(image_path)

        # Run the YOLOv8 model on the image
        results = model.predict(image)[0]

        # Draw the bounding boxes and labels on the image
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = model.names[int(cls)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Save the annotated image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")