"""SD3.5-medium with a domain LoRA, and the target-value embedding that conditions it on c.

The requested value c is standardized, (c - mu_R) / sigma_R, passed through a sinusoidal
projection and a two-layer MLP, and added to SD3's combined timestep / pooled-text embedding,
which conditions every transformer block. The MLP's last layer starts at zero, so training starts
from the unmodified model. Both classifier-free-guidance branches receive the same value embedding.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from .config import BASE_MODEL, BASE_REVISION

# Attention projections of the SD3 transformer that receive the LoRA.
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]


class ValueEmbedding(nn.Module):
    """Replaces ``transformer.time_text_embed``. While ``value`` (the standardized requested
    value) is set, its embedding is added to the timestep / pooled-text conditioning."""

    def __init__(self, base: nn.Module, embedding_dim: int):
        super().__init__()
        self.base = base
        self.projection = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        nn.init.zeros_(self.embedder.linear_2.weight)
        nn.init.zeros_(self.embedder.linear_2.bias)
        self.value: torch.Tensor | None = None   # shape (batch,) or (1,)

    def forward(self, timestep, pooled_projection):
        conditioning = self.base(timestep, pooled_projection)
        if self.value is None:
            return conditioning
        projected = self.projection(self.value).to(conditioning.dtype)
        return conditioning + self.embedder(projected)


def attach_value_embedding(transformer) -> ValueEmbedding:
    """Install a new ValueEmbedding on an SD3 transformer and return it."""
    width = transformer.time_text_embed.timestep_embedder.linear_2.out_features
    embedding = ValueEmbedding(transformer.time_text_embed, width)
    transformer.time_text_embed = embedding
    return embedding


def set_target(embedding: ValueEmbedding, target: float | None, stats: dict, device="cuda") -> None:
    """Set the requested value c on the embedding, or remove it with ``None``."""
    embedding.value = None if target is None else torch.tensor(
        [(float(target) - stats["mean"]) / stats["std"]], device=device, dtype=torch.float32)


def encode_prompt(prompt: str, device="cuda", dtype=torch.bfloat16):
    """``((cond, cond_pooled), (uncond, uncond_pooled))`` for one prompt. The text encoders are
    loaded for this call only."""
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE_MODEL, revision=BASE_REVISION, torch_dtype=dtype,
                                                    transformer=None, vae=None).to(device)
    with torch.no_grad():
        cond, uncond, pooled, unpooled = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=True)
    del pipe
    torch.cuda.empty_cache()
    return (cond, pooled), (uncond, unpooled)


def load_vae(device="cuda", dtype=torch.bfloat16):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(BASE_MODEL, revision=BASE_REVISION, subfolder="vae",
                                        torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()   # guidance back-propagates through the decoder
    return vae


def load_scheduler():
    return FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                                                           subfolder="scheduler")


def load_base_transformer(dtype=torch.bfloat16):
    from diffusers import SD3Transformer2DModel

    return SD3Transformer2DModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION, subfolder="transformer",
                                                 torch_dtype=dtype)


def load_scoredial(weights: str | Path, device="cuda", dtype=torch.bfloat16):
    """ScoreDial's generator: the base transformer with its LoRA adapter and value embedding.

    ``weights`` is a directory written by ``train.train``: the PEFT adapter
    (``adapter_model.safetensors``, ``adapter_config.json``) plus ``value_embedding.safetensors``
    and ``value_embedding.json``. Returns ``(transformer, embedding, stats)``, where ``stats``
    holds the mean and SD used to standardize c.
    """
    from peft import PeftModel
    from safetensors.torch import load_file

    weights = Path(weights)
    transformer = PeftModel.from_pretrained(load_base_transformer(dtype), str(weights)).to(device, dtype).eval()
    transformer.requires_grad_(False)
    embedding = attach_value_embedding(transformer.base_model.model)
    embedding.embedder.load_state_dict(load_file(weights / "value_embedding.safetensors"))
    embedding.to(device, dtype).eval().requires_grad_(False)
    stats = json.loads((weights / "value_embedding.json").read_text())
    return transformer, embedding, stats


def load_baseline(weights: str | Path, device="cuda", dtype=torch.bfloat16):
    """A baseline's generator: the base transformer with the baseline's LoRA merged in."""
    from peft import PeftModel

    transformer = PeftModel.from_pretrained(load_base_transformer(dtype), str(weights))
    transformer = transformer.to(device, dtype).merge_and_unload().eval()
    transformer.requires_grad_(False)
    return transformer
