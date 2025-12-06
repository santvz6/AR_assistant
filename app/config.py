import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
IMAGES_DIR = os.path.join(RESOURCES_DIR, "images")

VIDEOS_DIR = os.path.join(RESOURCES_DIR, "videos")
ENVIRONMENT_DIR = os.path.join(VIDEOS_DIR, "environment")

AUDIOS_DIR = os.path.join(RESOURCES_DIR, "audios")
ANSWERS_DIR = os.path.join(AUDIOS_DIR, "answers")
QUESTIONS_DIR = os.path.join(AUDIOS_DIR, "questions")

VIDEO_FPS = 30
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SEED = 1423454343

# Características
OBJECT_CLUES = {
    "color": [
        "rojo", "azul", "verde", "amarillo", "naranja", "morado", "rosa",
        "negro", "blanco", "gris", "cian", "magenta", "turquesa", "lila",
        "marrón", "beige", "dorado", "plateado"],
    
    "tamaño": [
        "minúsculo", "muy pequeño", "pequeño", 
        "mediano", "grande", "muy grande"],
    
    "categoria": [
        "animal", "fruta", "vehículo", "mueble", "electrónico",
        "ropa", "herramienta", "instrumento musical", "utensilio de cocina",
        "juguete", "deporte", "libro", "planta", "accesorio",
        "edificio", "transporte", "bebida", "comida", "material", "decoración"]
}


GAME_EXPLANATION = """
¡Bienvenido al juego interactivo! Te voy a guiar paso a paso. 
Primero, vamos a escanear los alrededores para identificar varios objetos y elementos en el escenario. 
Tu objetivo es identificar correctamente el objeto principal que hemos detectado usando nuestro sistema de visión.

Durante el juego, podrás hacer preguntas sobre lo que ves en el video, y yo te responderé según la información 
del objeto que hemos seleccionado. Cada cierto tiempo, te daré pistas para ayudarte a encontrar el objeto correcto. 
Las pistas pueden referirse al color, tamaño o categoría del objeto.

Al final, podrás confirmar si has identificado correctamente el objeto principal y recibirás retroalimentación inmediata.

¡Vamos a empezar y que te diviertas explorando el escenario!
"""