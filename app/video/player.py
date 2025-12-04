# app/video/player.py

import os
import numpy as np

from moviepy import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageDraw

from .detection import YOLODetector
from config import VIDEO_FPS

class VideoPlayer:
    def __init__(self, video_path, skip_seconds=1):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
        self.detector = YOLODetector()
        self.skip_seconds = skip_seconds

    def iter_frames(self):
        fps = VIDEO_FPS if VIDEO_FPS else self.clip.fps
        skip_frames = max(1, int(fps * self.skip_seconds))
        for i, frame in enumerate(self.clip.iter_frames(fps=fps)):
            if i % skip_frames == 0:
                yield frame

    def detect_objects(self, frame):
        """
        Detecta objetos y devuelve resultados.
        """
        return self.detector.detect(frame)

    def annotate_frame(self, frame, detections):
        """
        Dibuja bounding boxes y etiquetas sobre el frame
        """
        frame_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_img)
        for box, name, score in zip(detections.boxes.xyxy, detections.names, detections.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{name} {score:.2f}", fill="yellow")
        return np.array(frame_img)

    def process_video(self, output_path="resources/videos/output.mp4"):
        """
        Procesa todo el video: detección + anotación + creación de nuevo video
        
        
        """
        annotated_frames = []
        for frame in self.iter_frames():
            detections = self.detect_objects(frame)
            frame_annotated = self.annotate_frame(frame, detections)
            annotated_frames.append(frame_annotated)

        out_clip = ImageSequenceClip(annotated_frames, fps=self.clip.fps)
        out_clip.write_videofile(output_path)
        return output_path
