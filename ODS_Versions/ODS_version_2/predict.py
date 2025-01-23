import os
from PIL import Image
import cv2
from IPython.display import display
from ultralytics import YOLO

# Load a model
model = YOLO('/home/thales1/ODSv02/pythonProject/odsv02/runs/detect/train8/weights/best.pt')  # load a custom model

# Define the directory containing images
image_dir = ('/home/thales1/Hitachi_Repo/Obstacle_Detection_System/datasetsample/test/images')

# Iterate over the images in the directory
for image_file in os.listdir(image_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Predict with the model
    image_path = os.path.join(image_dir, image_file)
    results = model(image_path)  # predict on an image

    # Display the image with bounding boxes
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ensure results is a list of predictions
    if isinstance(results, list):
        for result in results:
            # Check the structure of result
            print("Result keys:", result.keys())

            # Access the labels, scores, and bounding boxes
            labels = result.get('labels')
            scores = result.get('scores')
            boxes = result.get('boxes')

            if labels is not None and scores is not None and boxes is not None:
                for label, conf, bbox in zip(labels, scores, boxes):
                    bbox = [int(coord) for coord in bbox]
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(image, f'{label} {conf:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

    # Convert image to PIL format for displaying in Jupyter Notebook
    pil_image = Image.fromarray(image)
    display(pil_image)
