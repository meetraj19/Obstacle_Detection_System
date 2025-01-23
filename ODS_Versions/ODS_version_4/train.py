from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.yaml').load('/home/thales1/Downloads/yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs= 2, imgsz=640,device = 'cpu')