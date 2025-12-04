import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
VIDEOS_DIR = os.path.join(RESOURCES_DIR, "videos")
IMAGES_DIR = os.path.join(RESOURCES_DIR, "images")

AUDIOS_DIR = os.path.join(RESOURCES_DIR, "audios")
ANSWERS_DIR = os.path.join(AUDIOS_DIR, "answers")
QUESTIONS_DIR = os.path.join(AUDIOS_DIR, "questions")

VIDEO_FPS = 30
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SEED = 1423443

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
