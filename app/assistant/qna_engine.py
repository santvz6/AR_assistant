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
        
        self.PAIR_QUEST_PER_CLUE = 2

    def start_loop(self, detected_object_path:str, detected_object_name:str, detected_object_names:str, detected_object_names_len:int) -> None:
        logger.debug(f"QnAEngine: start_loop(): starting loop...")
        questions_files = os.listdir(SPANISH_QUESTIONS_DIR)
        question_list = []
        question_idx = 0


        def process_question() -> str:
            """Genera el texto del audio utilizando Whisper"""
            logger.debug(f"QnAEngine: process_question(): question {question_idx}")
            question = self.audio.listen(os.path.join(SPANISH_QUESTIONS_DIR, questions_files[question_idx]))
            logger.debug(f"QnAEngine: process_question(): question: {question}")
            return clean_text(question)

        def process_answer() -> None:
            """Genera una respuesta basada en una Pareja de Preguntas utilizando CLIP"""
            answer_idx = question_idx // 2
            logger.debug(f"QnAEngine: process_answer(): answer {answer_idx + 1}")

            # Procesamiento con CLIP
            accuracies = self.clip.match(
                image_path=os.path.join(IMAGES_DIR, detected_object_path),
                text_list=question_list
            ).items()
            
            best_answer, best_score = max(accuracies, key=lambda x:x[1]) # mayor accuracy item
            question_list.clear()

            self.audio.speak(best_answer, filename=f"answer_{answer_idx+1}.mp3")
            logger.debug(f"QnAEngine: process_answer(): answer: {best_answer}")


        def process_clue() -> None:
            """Cada tres parejas de preguntas genera una pista utilizando CLIP"""    
            clue_keys = list(OBJECT_CLUES.keys())
            clue_index = ((question_idx + 1) // 2) // self.PAIR_QUEST_PER_CLUE
            clue_key = clue_keys[(clue_index-1) % len(clue_keys)]
            logger.debug(f"QnAEngine: process_clue(): clue key: {clue_key}")

            if clue_key == "objects":
                clue_text = (f"Aquí tienes una pista, atento. A continuación te dire los {detected_object_names_len} posibles objetos: {detected_object_names}")

            else:
                prob_dict = self.clip.match(
                    image_path=os.path.join(IMAGES_DIR, detected_object_path),
                    text_list=OBJECT_CLUES[clue_key]
                )

                best_key = max(prob_dict, key=prob_dict.get)
                best_score = int(prob_dict[best_key] * 100)
                clue_text = (f"Aquí tienes una pista, atento. {clue_key} {best_key} con un {best_score}% de confianza.")
            
            logger.debug(f"QnAEngine: process_clue(): clue_text: {clue_text}")
            self.audio.speak(clue_text, filename=f"clue_{question_idx//3 + 1}.mp3")
        


        # En unas gafas de realidad virtual sería un bucle cte
        while question_idx < len(questions_files):
            user_text = process_question() 
            if detected_object_name in user_text.split():
                logger.debug(f"QnAEngine: while question_idx < len(questions_files): success {detected_object_name}")
                self.audio.speak(f"¡Has acertado! El objeto era {detected_object_name}", filename="success.mp3")
                break
            else:
                question_list.append(user_text)

            # Cada dos Preguntas procesamos una Respuesta
            if ((question_idx + 1) % 2 == 0):
                process_answer()

                if ((question_idx + 1)//2) % self.PAIR_QUEST_PER_CLUE == 0:
                    process_clue()
                
            question_idx += 1
