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
from threading import Thread
from queue import Queue

from audio.listener import AudioListener
from audio.tts import TextToSpeech
from video.player import VideoPlayer
from game import Detection, GameLogic
from config import VIDEOS_DIR, VIDEO_FPS


def main():
    audio_queue = Queue()
    command_queue = Queue()

    # Variables
    seed = 153423443
    video_path = os.path.join(VIDEOS_DIR, os.listdir(VIDEOS_DIR)[0])

    # Inicializamos módulos
    listener = AudioListener(queue=audio_queue)
    tts = TextToSpeech()
    player = VideoPlayer(video_path=video_path, skip_seconds=0.01)
    game = GameLogic(command_queue=command_queue)

    # Hilo de audio
    Thread(target=listener.run, daemon=True).start()
 
    ############################################################################

    # 1. Esperar Orden Usuario
    #input("Iniciar juego: ") # input de audio

    # 2. Explicaión del juego
    if game.has_instruction():
            tts.say(game.get_instruction())

    # 3. Inicio del Juego
    ## 3.1 Video Processing
    output_video = player.process_video(output_path=os.path.join(VIDEOS_DIR, "video_annotated.mp4"))

    ## 3.2 Object Detection
    possible_detections = []
    possible_frames = []

    for frame in player.iter_frames():
        detections = player.detect_objects(frame)
        
        if detections:
            possible_detections.append(detections)
            possible_frames.append(frame)

    # Selección de frame y detecciones
    random.seed(seed)
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

    # Guardar frame anotado como PNG
    img_with_box = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_with_box.save(os.path.join(VIDEOS_DIR, "detected_frame.png"))

    # Imprimir detecciones
    print("\n ----------------------- OBJETOS DETECTADOS -----------------------")
    for cls in classes:
        print("Detección:", target_detections.names[int(cls)])


    print("\n ----------------------- OBJETIVO DETECTADO -----------------------")
    idx = random.randint(0, len(boxes) - 1)
    detection = Detection(boxes[idx], confs[idx], classes[idx], target_detections.names[int(cls)])



    # 4. Esperar preguntas del usuario
    while not audio_queue.empty():
        comando = audio_queue.get()
        game.process_command(comando)

        

if __name__ == "__main__":
    main()
