# app/main.py

# 1. Esperar Orden del Usuario: Speech2Text (Whisper)
    # 1.1 Transcribir audio: obtener orden.
    # 1.2 Analizar la orden (ej: contiene palabras como 'buscar', 'marcar', 'qué hay...').
# 2. Explicación del juego: Text2Speech
# 3. Inicio del juego 
    # 3.1 Video Processing
    # 3.2 Object Detection: YOLO
    # 3.3 Quedarnos con el Frame de la detección
# 4. Esperar preguntas del usuario
    # 4.1 Procesar respuestas: CLIP
    # 4.2 Quedarnos con el Frame de la respuesta
# 5. Cada cierto tiempo el Agente dará pistas
    # 5.1 ¿Comparando la puntuacion de cada palabra de nuestro dict clave?

import os
import random
import cv2

from PIL import Image

from audio import AudioListener, TextToSpeech
from video import VideoPlayer
from game import Detection, CLIPProcessor
from utils import clean_text
from config import VIDEOS_DIR, IMAGES_DIR, ANSWERS_DIR, QUESTIONS_DIR, AUDIOS_DIR
from config import OBJECT_CLUES
from config import SEED


class Main:
    def __init__(self):
        self.audio = AudioListener()
        self.tts = TextToSpeech(output_dir=ANSWERS_DIR)
        self.player = VideoPlayer(video_path=os.path.join(VIDEOS_DIR, os.listdir(VIDEOS_DIR)[0]), skip_seconds=0.01)
        self.clip = CLIPProcessor()

        random.seed(SEED)

    def __call__(self):
        # Pipline
        init = self.wait_user_init()
        if not init: return

        self.explain_game()
        self.process_enviorment()
        self.wait_user_questions()

    def wait_user_init(self):
        keywords = ["empezar", "iniciar", "vamos", "comenzar", "iniciemos", "comencemos", "juego", "escondite"]
        text = self.audio.process_audio(os.path.join(AUDIOS_DIR, "user_init.mp3"))

        for word in clean_text(text).split():
            if word in keywords:
                return True
            
        return False

    def explain_game(self):
        
        self.tts.say(
            filename="explain_game.mp3",
            texto="""
                ¡Bienvenido al juego interactivo! Te voy a guiar paso a paso. 
                Primero, vamos a escanear los alrededores para identificar varios objetos y elementos en el escenario. 
                Tu objetivo es identificar correctamente el objeto principal que hemos detectado usando nuestro sistema de visión.

                Durante el juego, podrás hacer preguntas sobre lo que ves en el video, y yo te responderé según la información 
                del objeto que hemos seleccionado. Cada cierto tiempo, te daré pistas para ayudarte a encontrar el objeto correcto. 
                Las pistas pueden referirse al color, tamaño o categoría del objeto.

                Al final, podrás confirmar si has identificado correctamente el objeto principal y recibirás retroalimentación inmediata.

                ¡Vamos a empezar y que te diviertas explorando el escenario!
            """)


    def process_enviorment(self):
        ## 3.1 Video Processing
        self.player.process_video(output_path=os.path.join(VIDEOS_DIR, "video_annotated.mp4"))

        ## 3.2 Object Detection
        possible_detections = []
        possible_frames = []

        for frame in self.player.iter_frames():
            detections = self.player.detect_objects(frame)
            
            if detections:
                possible_detections.append(detections)
                possible_frames.append(frame)

        # Selección de frame y detecciones
        
        target_index = random.choice(list(range(len(possible_detections))))

        target_frame = possible_frames[target_index]
        target_detections = possible_detections[target_index]  # Results object de YOLOv8

        # Convertir frame a BGR para OpenCV
        frame_bgr = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)

        # Convertir tensores a NumPy
        boxes = target_detections.boxes.xyxy.cpu().numpy()   # (N,4)
        confs = target_detections.boxes.conf.cpu().numpy()   # (N,)
        classes = target_detections.boxes.cls.cpu().numpy()  # (N,)

        # Dibujar recuadros y etiquetas
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = target_detections.names[int(cls)]
            color = (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Guardar frame detectado como PNG
        img_with_box = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img_with_box.save(os.path.join(IMAGES_DIR, "detected_frame.png"))
            
        # Imprimir detecciones
        print("\n ----------------------- OBJETOS DETECTADOS -----------------------")
        for cls in classes:
            print("Detección:", target_detections.names[int(cls)])


        print("\n ----------------------- OBJETIVO DETECTADO -----------------------")
        weights = confs / confs.sum()  # convierte a probabilidades
        best_det_idx = random.choices(range(len(boxes)), weights=weights, k=1)[0]

        # Crear el objeto Detection
        detection = Detection(
            boxes[best_det_idx],
            confs[best_det_idx],
            classes[best_det_idx],
            target_detections.names[int(classes[best_det_idx])]
        )
        print(detection.label)

        # Guardar objeto detectado como PNG
        x1, y1, x2, y2 = map(int, boxes[best_det_idx])
        
        margin = 20
        h, w, _ = target_frame.shape
        x1m = max(x1 - margin, 0)
        y1m = max(y1 - margin, 0)
        x2m = min(x2 + margin, w)
        y2m = min(y2 + margin, h)
        cropped_object = target_frame[y1m:y2m, x1m:x2m]

        cropped_pil = Image.fromarray(cropped_object)
        cropped_pil.save(os.path.join(IMAGES_DIR, "detected_object.png"))
        

    def wait_user_questions(self):
        questions_files = os.listdir(QUESTIONS_DIR)
        question_idx = 0

        def process_question():
            return self.audio.process_audio(os.path.join(QUESTIONS_DIR, questions_files[question_idx]))

        def process_answer(user_text):
            user_clean_text = clean_text(user_text)

            img_emb = self.clip.get_image_embedding(image_path=os.path.join(IMAGES_DIR, "detected_object.png"))
            text_emb = self.clip.get_text_embedding(text=user_clean_text)
            accuracy = self.clip.cosine_similarity(img_emb, text_emb)

            print(user_clean_text)
            THRESHOLD = 0.25
            if accuracy >= THRESHOLD:
                answer = f"Sí, es correcto. ({accuracy*100:.0f}%)"
            else:
                answer = f"No estoy seguro. ({accuracy*100:.0f}%)"

            print(answer)
            self.tts.say(answer, filename=f"answer_{question_idx+1}.mp3")

        def process_clue():
            # Assistant Clue cada 3 respuestas
            if (question_idx + 1) % 3 == 0:
                clue_keys = list(OBJECT_CLUES.keys())
                clue_key = clue_keys[(question_idx // 3) % len(clue_keys)]

                prob_dict = self.clip.match(
                    image_path=os.path.join(IMAGES_DIR, "detected_frame.png"),
                    text_list=OBJECT_CLUES[clue_key]
                )
                sorted_keys = sorted(prob_dict, key=lambda x: prob_dict[x], reverse=True)
                self.tts.say(
                    texto=f"Aquí tienes una pista, atento. {clue_key} {sorted_keys[0]} con un {int(prob_dict[sorted_keys[0]] * 100)}% de confianza", 
                    filename=f"clue_{question_idx // 3}.mp3")

        while question_idx < len(questions_files):
            user_text = process_question()
            process_answer(user_text)
            process_clue()
            question_idx += 1



if __name__ == "__main__":
    main = Main()
    main()