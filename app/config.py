import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
AUDIOS_DIR = os.path.join(RESOURCES_DIR, "audios")
VIDEOS_DIR = os.path.join(RESOURCES_DIR, "videos")
FRAMES_OUT_DIR = os.path.join(RESOURCES_DIR, "frames_out")

VIDEO_FPS = 30
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
