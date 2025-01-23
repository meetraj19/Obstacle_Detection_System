from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml').load('../../models/yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs= 75, imgsz=640)