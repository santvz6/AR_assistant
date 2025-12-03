# app/utils/helpers.py

import os
import time

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")
