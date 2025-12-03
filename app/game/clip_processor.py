# app/game/clip_processor.py

import clip
import torch
from PIL import Image

from app.config import DEVICE

class CLIPProcessor:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)

    def match(self, ruta_imagen, lista_textos):
        imagen = self.preprocess(Image.open(ruta_imagen)).unsqueeze(0).to(DEVICE)
        textos = clip.tokenize(lista_textos).to(DEVICE)
        with torch.no_grad():
            logits_imagen, logits_texto = self.model(imagen, textos)
            probs = logits_imagen.softmax(dim=-1).cpu().numpy()[0]
        return dict(zip(lista_textos, probs))
