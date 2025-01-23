import os
import cv2
from ultralytics import YOLOWorld

# Initialize the model with the specified weights file
model = YOLOWorld('../../models/yolov8m-world.pt')

# Set the classes for the model. This is just an example; adjust based on your model's documentation and requirements.
model.set_classes(['animal', 'bicycle', 'bus-truck', 'car', 'other-vehicles', 'person', 'train',
                    'Signal-Green', 'Signal-Yellow', 'Signal-red',
                   'Station-Lights-Off', 'Station-lights-ON', 'bridge', 'building', 'cementblock', 'dwarf signal',
                   'electric pole', 'fencing', 'fire', 'house', 'instruction sign', 'pillar', 'plasticbag',
                   'platform', 'railwaytrack', 'smoke', 'stairs', 'surveillance', 'traffic signal',
                   'tree', 'wall'])

# Directory containing the images
directory = ('/home/thales1/Hitachi_Repo/Obstacle_Detection_System/datasetsample/test/images'
             )

# Process and display each image in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        image_path = os.path.join(directory, filename)
        try:
            # Load the image
            img = cv2.imread(image_path)

            # Predict using the model
            results = model.predict(source=img, save=False)

            # Process the results
            for result in results:
                boxes = result.boxes.data.tolist()  # Get bounding box coordinates
                labels = result.boxes.cls.data.tolist()  # Get class labels
                scores = result.boxes.conf.data.tolist()  # Get confidence scores

                for box, label, score in zip(boxes, labels, scores):
                    if len(box) == 4:  # Check if the box has four elements
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                    else:
                        # Handle the case when the box has more than four elements
                        # For example, you can take the first four elements or skip the box
                        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]  # Take the first four elements

                    class_name = model.names[label]
                    confidence = f"{score * 100:.2f}%"

                    # Draw bounding box and label on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{class_name} {confidence}"
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

            # Display the image with bounding boxes and labels using OpenCV
            cv2.imshow("Object Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing {filename}: {e}")