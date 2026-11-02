"""The attribute regressor R: a frozen pretrained image encoder followed by a ridge head.

The same differentiable preprocessing is used when fitting R and when R guides sampling:
RGB in [0, 1], antialiased bicubic resize of the shorter side to 256, 224x224 centre crop,
then the encoder's own normalization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

BACKBONES = {
    "dinov2": dict(model_id="facebook/dinov2-base",
                   revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "dinov3": dict(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   revision="5931719e67bbdb9737e363e781fb0c67687896bc",
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "clip": dict(model_id="openai/clip-vit-base-patch16",
                 revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
                 mean=(0.48145466, 0.4578275, 0.40821073),
                 std=(0.26862954, 0.26130258, 0.27577711)),
}
# Backbone:representation pairs considered for R. On equal scores the earlier entry is chosen.
CANDIDATES = (
    "dinov2:patchmean", "dinov2:cls", "dinov2:cls+patchmean",
    "dinov3:patchmean", "dinov3:cls", "dinov3:cls+patchmean",
    "clip:projected",
)
RIDGE_ALPHAS = (1.0, 10.0, 50.0, 100.0)


def load_encoder(family: str, device):
    from transformers import AutoModel, CLIPModel

    spec = BACKBONES[family]
    model_class = CLIPModel if family == "clip" else AutoModel
    return model_class.from_pretrained(spec["model_id"], revision=spec["revision"]).to(device).eval()


def preprocess(image01: torch.Tensor) -> torch.Tensor:
    """Resize and crop (B, 3, H, W) images in [0, 1] to (B, 3, 224, 224); differentiable."""
    height, width = image01.shape[-2:]
    scale = 256.0 / min(height, width)
    image01 = F.interpolate(image01, size=(round(height * scale), round(width * scale)),
                            mode="bicubic", align_corners=False, antialias=True)
    height, width = image01.shape[-2:]
    top, left = (height - 224) // 2, (width - 224) // 2
    return image01[:, :, top:top + 224, left:left + 224]


def encode(encoder, family: str, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every representation an encoder offers, from normalized pixels."""
    if family == "clip":
        vision = encoder.vision_model(pixel_values=pixels)
        return {"projected": encoder.visual_projection(vision.pooler_output)}
    tokens = encoder(pixel_values=pixels).last_hidden_state
    registers = int(getattr(encoder.config, "num_register_tokens", 0) or 0)   # skipped in the patch mean
    return {"cls": tokens[:, 0], "patchmean": tokens[:, 1 + registers:].mean(1)}


def representation(features: dict, pooling: str):
    """One named representation; ``cls+patchmean`` concatenates the two."""
    if pooling in features:
        return features[pooling]
    parts = [features[part] for part in pooling.split("+")]
    return torch.cat(parts, dim=1) if torch.is_tensor(parts[0]) else np.concatenate(parts, axis=1)


def _read_image01(path: str, device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def extract(family: str, paths: list[str], device="cuda", batch_size: int = 16) -> dict[str, np.ndarray]:
    """Features of image files under every representation of one encoder."""
    encoder = load_encoder(family, device)
    mean = torch.tensor(BACKBONES[family]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(BACKBONES[family]["std"], device=device).view(1, 3, 1, 1)
    batches: dict[str, list] = {}
    for start in range(0, len(paths), batch_size):
        # Each image is resized on its own because sizes can differ within a dataset.
        pixels = torch.cat([preprocess(_read_image01(p, device)) for p in paths[start:start + batch_size]])
        for name, value in encode(encoder, family, (pixels - mean) / std).items():
            batches.setdefault(name, []).append(value.float().cpu().numpy())
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {name: np.concatenate(values).astype(np.float32) for name, values in batches.items()}


class Regressor(nn.Module):
    """R(x): images in [-1, 1] to attribute scores on [0, 1], differentiable with respect to the image."""

    def __init__(self, head: dict, device="cuda"):
        super().__init__()
        self.family, self.pooling = head["backbone"], head["pooling"]
        self.encoder = load_encoder(self.family, device)
        self.encoder.requires_grad_(False)
        spec = BACKBONES[self.family]
        self.register_buffer("image_mean", torch.tensor(spec["mean"]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(spec["std"]).view(1, 3, 1, 1))
        self.register_buffer("feat_mean", torch.as_tensor(np.asarray(head["feat_mean"]), dtype=torch.float32))
        self.register_buffer("feat_scale", torch.as_tensor(np.asarray(head["feat_scale"]), dtype=torch.float32))
        self.register_buffer("weight", torch.as_tensor(np.asarray(head["weight"]), dtype=torch.float32))
        self.bias = float(head["bias"])
        self.to(device)

    def features(self, image_m1_1: torch.Tensor) -> torch.Tensor:
        pixels = preprocess((image_m1_1.clamp(-1, 1) + 1) / 2)
        pixels = (pixels - self.image_mean) / self.image_std
        return representation(encode(self.encoder, self.family, pixels), self.pooling)

    def forward(self, image_m1_1: torch.Tensor) -> torch.Tensor:
        standardized = (self.features(image_m1_1) - self.feat_mean) / self.feat_scale
        return standardized @ self.weight + self.bias

    @classmethod
    def load(cls, path, device="cuda") -> "Regressor":
        return cls(load_head(path), device=device)


def save_head(head: dict, path) -> None:
    """Write a ridge head (backbone, pooling, alpha, feature standardization, weights) as JSON."""
    serial = {k: (np.asarray(v).astype(float).tolist() if isinstance(v, np.ndarray) else v)
              for k, v in head.items()}
    Path(path).write_text(json.dumps(serial))


def load_head(path) -> dict:
    return json.loads(Path(path).read_text())
