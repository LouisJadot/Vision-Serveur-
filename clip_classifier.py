import json
import torch
import numpy as np
import cv2
import open_clip
from PIL import Image

from config import DEVICE, CLIP_MODEL, CLIP_WEIGHTS, DEFAULT_LABELS, LABELS_PATH


class ClipClassifier:

    def __init__(self):

        self.device = DEVICE

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL,
            pretrained=CLIP_WEIGHTS,
        )

        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model = self.model.to(self.device).eval()

        self.labels = self._load_labels()
        self._encode_text()


    def _load_labels(self):
        try:
            with open(LABELS_PATH, "r") as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                print(f"Labels chargés: {data}")
                return data

        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return list(DEFAULT_LABELS)


    def _save_labels(self):
        try:
            with open(LABELS_PATH, "w") as f:
                json.dump(self.labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur save labels: {e}")


    def update_labels(self, new_labels):

        cleaned = []

        for l in new_labels:
            if isinstance(l, str) and l.strip():
                cleaned.append(l.strip())

        cleaned = list(dict.fromkeys(cleaned))

        if len(cleaned) == 0:
            print("⚠️ fallback DEFAULT_LABELS")
            cleaned = list(DEFAULT_LABELS)

        self.labels = cleaned
        self._encode_text()
        self._save_labels()

    def reset(self):
        self.labels = list(DEFAULT_LABELS)
        self._encode_text()
        self._save_labels()

    def get_labels(self):
        return self.labels

    def _encode_text(self):
        tokens = self.tokenizer(self.labels).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)

        self.text_features = feats

    def predict(self, crop_bgr: np.ndarray):

        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

        except Exception as e:
            print(f"CLIP preprocess error: {e}")
            return "unknown", 0.0

        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ self.text_features.T).squeeze(0)

        idx = similarity.argmax().item()
        score = similarity[idx].item()

        return self.labels[idx], score