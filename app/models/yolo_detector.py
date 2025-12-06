# app/mdodels/yolo_detector.py

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame_or_image_path):
        resultados = self.model(frame_or_image_path)
        #resultados[0].show()  # muestra con bounding boxes (el problema es que abre GIMP Muchas veces y ni se muestra bien)
        return resultados[0]
