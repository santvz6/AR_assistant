# app/mdodels/clip_processor.py

import clip
import torch
from PIL import Image
import torch.nn.functional as F

from config import DEVICE
from logger_config import logger

class CLIPProcessor:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)


    def match(self, image_path, text_list):
        """Compara una imagen con un texto y devuelve una lista de tuplas (texto, probabilidad)."""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        texts = clip.tokenize(text_list).to(DEVICE)

        with torch.no_grad():
            image_logits, text_logits = self.model(image, texts)
            probs = image_logits.softmax(dim=-1).cpu().numpy()[0]
        return dict(zip(text_list, probs))