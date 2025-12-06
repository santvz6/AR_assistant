# app/assistant/user_init.py

import os

from manager import AudioManager
from utils import clean_text
from config import AUDIOS_DIR


class UserInitHandler:
    KEYWORDS = ["empezar", "iniciar", "vamos", "comenzar", "iniciemos", "comencemos", "juego", "escondite"]

    def __init__(self):
        self.audio = AudioManager()

    def wait_for_init(self):
        text = self.audio.listen(os.path.join(AUDIOS_DIR, "user_init.mp3"))

        for word in clean_text(text).split():
            if word in self.KEYWORDS:
                return True
        return False
