# app/assistant/qna_engine.py

import os

from manager import AudioManager
from models import CLIPProcessor
from utils import clean_text
from config import SPANISH_QUESTIONS_DIR, OBJECT_CLUES, IMAGES_DIR
from logger_config import logger

class QnAEngine:
    THRESHOLD = 0.25

    def __init__(self):
        self.audio = AudioManager()
        self.clip = CLIPProcessor()

    def start_loop(self, detected_object_path, detected_frame_path):
        logger.debug(f"QnAEngine: start_loop(): starting loop...")
        questions_files = os.listdir(SPANISH_QUESTIONS_DIR)
        question_list = []
        question_idx = 0


        def process_question():
            """Genera el texto del audio utilizando Whisper"""
            logger.debug(f"QnAEngine: process_question(): question {question_idx}")
            question = self.audio.listen(os.path.join(SPANISH_QUESTIONS_DIR, questions_files[question_idx]))
            logger.debug(f"QnAEngine: process_question(): question: {question}")
            return question

        def process_answer():
            answer_idx = question_idx // 2
            logger.debug(f"QnAEngine: process_answer(): answer {answer_idx}")

            accuracies = self.clip.match(
                image_path=os.path.join(IMAGES_DIR, detected_object_path),
                text_list=question_list
            ).items()
            
            best_answer, best_score = max(accuracies, key=lambda x:x[1]) # mayor accuracy
            question_list.clear()

            self.audio.speak(best_answer, filename=f"answer_{answer_idx+1}.mp3")
            logger.debug(f"QnAEngine: process_answer(): answer: {best_answer}")


        def process_clue():
            """Cada tres preguntas genera una pista utilizando CLIP"""    
            clue_keys = list(OBJECT_CLUES.keys())
            clue_key = clue_keys[(question_idx // 3) % len(clue_keys)]
            logger.debug(f"QnAEngine: process_clue(): clue key: {clue_key}")

            prob_dict = self.clip.match(
                image_path=os.path.join(IMAGES_DIR, detected_object_path),
                text_list=OBJECT_CLUES[clue_key]
            )

            best_key = max(prob_dict, key=prob_dict.get)
            best_score = int(prob_dict[best_key] * 100)

            clue_text = (f"Aquí tienes una pista, atento. {clue_key} {best_key} con un {best_score}% de confianza.")
            logger.debug(f"QnAEngine: process_clue(): clue_text: {clue_text}")
            self.audio.speak(clue_text, filename=f"clue_{question_idx//3}.mp3")

        # En unas gafas de realidad virtual sería un bucle
        while question_idx < len(questions_files):
            user_text = process_question()
            question_list.append(clean_text(user_text))
            
            if ((question_idx + 1) % 2 == 0):
                process_answer()

            if (max(1, (question_idx + 1)//2)) % 3 == 0:
                process_clue()
                
            question_idx += 1
