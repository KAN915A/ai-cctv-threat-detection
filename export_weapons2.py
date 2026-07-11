from pathlib import Path

from ultralytics import YOLO

src = ("Weapons-and-Knives-Detector-with-YOLOv8-main/runs/detect/"
       "Normal_Compressed/weights/best.pt")
model = YOLO(src)
print("classes:", model.names)
out = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
Path(out).replace("web/models/weapons2.onnx")
print("saved web/models/weapons2.onnx")
