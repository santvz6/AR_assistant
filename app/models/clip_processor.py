# app/mdodels/clip_processor.py

import clip
import torch
from PIL import Image
import torch.nn.functional as F

from config import DEVICE


class CLIPProcessor:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)


    def match(self, image_path, text_list):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        texts = clip.tokenize(text_list).to(DEVICE)

        with torch.no_grad():
            image_logits, text_logits = self.model(image, texts)
            probs = image_logits.softmax(dim=-1).cpu().numpy()[0]
        return dict(zip(text_list, probs))


    def get_image_embedding(self, image_path):
        """Devuelve el embedding CLIP de la imagen."""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_emb = self.model.encode_image(image)
            img_emb = F.normalize(img_emb, p=2, dim=-1)  # normalización L2
        return img_emb.cpu().numpy()[0]


    def get_text_embedding(self, text):
        """Devuelve el embedding CLIP del texto."""
        tokens = clip.tokenize([text]).to(DEVICE)

        with torch.no_grad():
            txt_emb = self.model.encode_text(tokens)
            txt_emb = F.normalize(txt_emb, p=2, dim=-1)
        return txt_emb.cpu().numpy()[0]


    def cosine_similarity(self, vec1, vec2):
        """Similitud coseno entre dos embeddings."""
        v1 = torch.tensor(vec1)
        v2 = torch.tensor(vec2)
        return F.cosine_similarity(v1, v2, dim=0).item()
