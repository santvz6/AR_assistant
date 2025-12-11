# app/manager/video.py

import os
import random
import numpy as np

from moviepy import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

from models import YOLODetector
from config import VIDEO_FPS, IMAGES_DIR
from logger_config import logger


class VideoManager:
    def __init__(self, video_path, sample_percentage):
        self.video_path = video_path
        self.video_clip = VideoFileClip(video_path)
        self.detector = YOLODetector()
        self.sample_percentage = max(1, min(100, sample_percentage))

    def __iter_frames(self):
        """
        Iterea los frames de un vídeo utilizando VideoFileClip().
        """
        fps = VIDEO_FPS if VIDEO_FPS else self.video_clip.fps
        total_frames = int(self.video_clip.duration * fps)

        # Calculamos cada cuántos frames tomar según el porcentaje
        step = max(1, int(100 / self.sample_percentage))
        for i, frame in enumerate(self.video_clip.iter_frames(fps=fps)):
            if i % step == 0:
                yield frame

    def __detect_objects(self, frame):
        """
        Detecta objetos utilizando YOLO, devuelve los resultados.
        """
        return self.detector.detect(frame)

    def __annotate_frame(self, detections):
        """
        Usa el método nativo de YOLO para dibujar las cajas y etiquetas.
        """
        return detections.plot()


    def __save_annotated_frame(self, detections, filename="detected_frame.png"):
        """
        Utiliza __anotate_frame() y guarda la imagen.
        """
        annotated = self.__annotate_frame(detections)
        img = Image.fromarray(annotated)
        self.detected_frame_path = os.path.join(IMAGES_DIR, filename)
        img.save(self.detected_frame_path)


    def __save_select_detection(self, frame, detections, margin=-10, filename="detected_object.png"):
        """
        Guarda y selecciona una detección de forma aleatoria utilizando como pesos la confianza. 
        """
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()

        weights = confs / confs.sum()
        idx = random.choices(range(len(boxes)), weights=weights, k=1)[0]

        self.detected_object_name = detections.names[int(classes[idx])]

        logger.debug(f"VideoManager: process_video(): Selected Object -> "
                     f"Class: {self.detected_object_name}, Confidence: {confs[idx]:.2f}")

        x1, y1, x2, y2 = map(int, boxes[idx])
        h, w, _ = frame.shape

        crop = frame[
            max(0, y1-margin):min(h, y2+margin),    # y1:y2
            max(0, x1-margin):min(w, x2+margin)     # x1:x2
        ]

        # Guardamos la imagen
        self.detected_object_path = os.path.join(IMAGES_DIR, filename)
        Image.fromarray(crop).save(self.detected_object_path)
    

    def process_video(self, output_path="resources/videos/output.mp4"):
        """
        Procesa todos los frames del vídeo y almacena información relevante.
        """
        logger.debug("VideoManager: process_video(): processing_video...")
        annotated_frames = []
        dets_frames, dets = [], []

        self.detected_object_names = set()

        for frame in self.__iter_frames():
            # Detección de objetos
            detections = self.__detect_objects(frame)

            # Recorremos las detecciones del frame y añadir la etiqueta al conjunto
            if detections:
                for cls in detections.boxes.cls:
                    self.detected_object_names.add(detections.names[int(cls)])

            # Dibujo de boxes sobre el frame
            frame_annotated = self.__annotate_frame(detections)
            annotated_frames.append(frame_annotated)

            # Guardamos Frames con Objetos y Sus Objetos
            if detections:
                dets_frames.append(frame)
                dets.append(detections)


        # Recreamos el vídeo con las imagenes creadas
        out_clip = ImageSequenceClip(annotated_frames, fps=self.video_clip.fps)
        out_clip.write_videofile(output_path)

        # Selección Aleatoria de un Frame con sus Objetos
        idx = random.randrange(0, len(dets))
        logger.debug(f"VideoManager: process_video(): index: {idx}")

        target_frame = dets_frames[idx]
        target_dets   = dets[idx]

        for _, cls, score in zip(target_dets.boxes.xyxy, target_dets.boxes.cls, target_dets.boxes.conf):
            logger.debug(f"VideoManager: process_video(): Object Target Frame Detections -> "
                         f"Class: {detections.names[int(cls)]}, Confidence: {score:.2f}")
        

        # Guardamos las imagenes del Frame y del Objeto
        self.__save_annotated_frame(target_dets)
        self.__save_select_detection(target_frame, target_dets)
