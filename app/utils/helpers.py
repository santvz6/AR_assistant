# app/utils/helpers.py

import os
import time
import cv2


def get_video_frame_count_for_fps(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("No se pudo abrir el video")

   
    original_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Duración real del video
    duration = original_frames / original_fps

    # Cantidad de frames para el FPS deseado
    target_frames = int(duration * fps)

    return target_frames
