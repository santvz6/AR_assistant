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

from assistant import ARAssistant

if __name__ == "__main__":
    assistant = ARAssistant()
    assistant()
