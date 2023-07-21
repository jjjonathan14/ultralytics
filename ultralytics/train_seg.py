from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8l-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8l-seg.yaml').load('yolov8l.pt')  # build from YAML and transfer weights

model = YOLO('yolov8l.pt')

# Train the model
model.train(data='cfg/datasets/coco128-seg.yaml', epochs=300, imgsz=512, batch=8, dropout=0.1,resume=True)