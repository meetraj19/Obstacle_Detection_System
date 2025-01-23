from ultralytics import YOLO

# Load an official or custom model
model = YOLO('/home/thales1/ODSv02/pythonProject/odsv02/runs/detect/train11/weights/best.pt')  # Load a custom trained model

# Perform tracking with the model
results = model.track(source='/home/thales1/Test_Videos/istockphoto-1250036378-640_adpp_is.mp4', show=True)  # Tracking with default tracker
