# app/assistant/assistant.py

import os
import random
from googletrans import Translator

from .user_init import UserInitHandler
from .qna_engine import QnAEngine
from manager import VideoManager, AudioManager
from config import SEED, GAME_EXPLANATION
from config import ENVIRONMENT_DIR
from logger_config import logger

class ARAssistant:
    def __init__(self):
        random.seed(SEED)
        logger.info(f"SEED: {SEED}")

        video_file = next((f for f in os.listdir(ENVIRONMENT_DIR) if f == "environment.mp4"), None)
        logger.debug(f"video_file: {video_file}")
        self.video = VideoManager(
            video_path=os.path.join(ENVIRONMENT_DIR, video_file), 
            sample_percentage=100)
        
        self.user_init = UserInitHandler()
        self.audio = AudioManager()
        self.qna = QnAEngine()

        self.translator = Translator()

    def __call__(self):
        # pipline
        if not self.user_init.wait_for_init():
            return
        self.audio.speak(text=GAME_EXPLANATION, filename="explain_game.mp3")
        self.video.process_video(output_path=os.path.join(ENVIRONMENT_DIR, "processed_video.mp4"))
        self.qna.start_loop(
            detected_object_path=self.video.detected_object_path,   
            detected_object_name=self.translator.translate(self.video.detected_object_name, src="en", dest="es").text,
            detected_object_names=self.translator.translate(", ".join(self.video.detected_object_names), src="en", dest="es").text,
            detected_object_names_len=len(self.video.detected_object_names)
        )
