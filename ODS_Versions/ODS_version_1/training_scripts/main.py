from ultralytics import YOLO


#load a pretrained model

model  = YOLO("yolov8s.yaml")

#use this model

results = model.train(data="data.yaml",epochs=10)