import os
import torch
import numpy as np
import cv2


# Define the calculate_iou function
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate union area
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union
    return iou


# Load your custom-trained YOLOv8 model
model_path = '/home/thales1/ODSv02/pythonProject/runs/detect/train20/weights/best.pt'  # Path to your custom-trained YOLOv8 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(model_path, map_location=device)['model'].float()  # Load the model

# Define directories
input_dir = '/home/thales1/OSD4kdataset/test/images'  # Directory containing test images
annotations_dir = '/home/thales1/OSD4kdataset/test/labels'  # Directory containing ground truth annotations

# Define detection threshold (adjust as needed)
threshold = 0.5

# Initialize variables for evaluation
total_objects = 0
correct_objects = 0

# Iterate through test images
for image_file in os.listdir(input_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Read the image
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}. Skipping.")
        continue

    # Convert image to PyTorch tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0

    # Perform object detection
    results = model(image_tensor.unsqueeze(0))  # Add batch dimension

    # Print the structure of results
    print(results)

    # Extract detections
    if isinstance(results, tuple):
        results = results[0]  # Handle tuple output

    # Load ground truth annotations
    annotation_file = os.path.join(annotations_dir, f'{os.path.splitext(image_file)[0]}.txt')
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    # Extract ground truth objects
    ground_truth = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        class_id = int(class_id)
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])
        ground_truth.append({'class_id': class_id, 'bbox': [x1, y1, x2, y2]})

    # Compare detections with ground truth
    for detection in results[0]:  # Loop through detections
        pred_class_id = detection[5].argmax().item()  # Find the class with maximum confidence
        pred_confidence = detection[4].item()  # Extract confidence for the predicted class
        pred_bbox = detection[:4].detach().cpu().numpy().astype(np.int32)

        # Compare with ground truth
        for obj in ground_truth:
            if obj['class_id'] == pred_class_id:
                iou = calculate_iou(obj['bbox'], pred_bbox)
                if iou > 0.5 and pred_confidence > threshold:
                    correct_objects += 1
                    break

    total_objects += len(ground_truth)

# Compute accuracy
accuracy = correct_objects / total_objects if total_objects > 0 else 0
print(f"Total objects: {total_objects}")
print(f"Correct detections: {correct_objects}")
print(f"Accuracy: {accuracy:.2f}")
