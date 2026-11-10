"""Frozen SigLIP image embeddings."""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class Encoder:
    def __init__(self, model_id: str):
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def images(self, pil_images, batch: int = 32) -> np.ndarray:
        out = []
        for s in range(0, len(pil_images), batch):
            inputs = self.processor(images=[im.convert("RGB") for im in pil_images[s:s + batch]], return_tensors="pt")
            f = self.model.get_image_features(**inputs)
            f = getattr(f, "pooler_output", f)
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32)
