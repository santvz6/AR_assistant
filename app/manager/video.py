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
    def __init__(self, video_path, skip_seconds=1):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
        self.detector = YOLODetector()
        self.skip_seconds = skip_seconds

    def __iter_frames(self):
        fps = VIDEO_FPS if VIDEO_FPS else self.clip.fps
        skip_frames = max(1, int(fps * self.skip_seconds))
        for i, frame in enumerate(self.clip.iter_frames(fps=fps)):
            if i % skip_frames == 0:
                yield frame

    def __detect_objects(self, frame):
        """
        Detecta objetos y devuelve resultados.
        """
        return self.detector.detect(frame)

    def __annotate_frame(self, frame, detections, text_scale=2):
        """
        Dibuja bounding boxes y etiquetas sobre el frame
        """
        frame_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_img)
        for box, cls, score in zip(detections.boxes.xyxy, detections.boxes.cls, detections.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)    
            text_img = Image.new("RGBA", (100, 30), (0, 0, 0, 0))
            ImageDraw.Draw(text_img).text(xy=(0, 0), text=f"{detections.names[int(cls)]} {score:.2f}", fill="yellow")
            text_img = text_img.resize((text_img.width * text_scale, text_img.height * text_scale), Image.NEAREST)
            frame_img.paste(text_img, (x1, y1 - text_img.height), text_img)
        return np.array(frame_img)


    def __save_annotated_frame(self, frame, detections, filename="detected_frame.png"):
        annotated = self.__annotate_frame(frame, detections)
        img = Image.fromarray(annotated)
        self.detected_frame_path = os.path.join(IMAGES_DIR, filename)
        img.save(self.detected_frame_path)


    def __save_best_object_crop(self, frame, detections, margin=20, filename="detected_object.png"):
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()

        weights = confs / confs.sum()
        idx = random.choices(range(len(boxes)), weights=weights, k=1)[0]
        logger.debug(f"VideoManager: process_video(): Selected Object -> "
                     f"Class: {detections.names[int(classes[idx])]}, Confidence: {confs[idx]:.2f}")

        x1, y1, x2, y2 = map(int, boxes[idx])
        h, w, _ = frame.shape

        crop = frame[
            max(0, y1-margin):min(h, y2+margin),    # y1:y2
            max(0, x1-margin):min(w, x2+margin)     # x1:x2
        ]
        self.detected_object_path = os.path.join(IMAGES_DIR, filename)
        Image.fromarray(crop).save(self.detected_object_path)
    

    def process_video(self, output_path="resources/videos/output.mp4"):
        logger.debug("VideoManager: process_video(): processing_video...")
        annotated_frames = []
        dets_frames, dets = [], []
        for frame in self.__iter_frames():
            # Detección de objetos
            detections = self.detector.detect(frame)

            # Dibujo de boxes sobre el frame
            frame_annotated = self.__annotate_frame(frame, detections)
            annotated_frames.append(frame_annotated)

            # Guardamos Frames con Objetos y Sus Objetos
            if detections:
                dets_frames.append(frame)
                dets.append(detections)

        # Recreamos el vídeo con las imagenes creadas
        out_clip = ImageSequenceClip(annotated_frames, fps=self.clip.fps)
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
        self.__save_annotated_frame(target_frame, target_dets)
        self.__save_best_object_crop(target_frame, target_dets)
