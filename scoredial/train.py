"""Joint flow-matching training of a domain LoRA and the target-value embedding.

Every training image is pseudo-labelled by the deployed regressor R. The labels are standardized
by their mean and SD over the pool; both are saved with the embedding and reused when sampling.
With ``value_embedding=False`` the same recipe trains a LoRA alone (the baselines' adapters).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import BASE_MODEL, BASE_REVISION, TRAINING
from .model import LORA_TARGETS, attach_value_embedding

DTYPE = torch.bfloat16
BATCH_FOR_ENCODING = 8


def _image_transform(size: int):
    return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                               transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def _image_batches(paths: list[str], size: int):
    transform = _image_transform(size)
    for start in range(0, len(paths), BATCH_FOR_ENCODING):
        yield torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + BATCH_FOR_ENCODING]])


@torch.no_grad()
def pseudo_label(paths: list[str], regressor, size: int, device="cuda") -> np.ndarray:
    """R's scores for the training images, as seen at training resolution."""
    scores = [regressor(batch.to(device).float()).flatten().float().cpu().numpy()
              for batch in _image_batches(paths, size)]
    return np.concatenate(scores).astype(np.float32)


@torch.no_grad()
def _encode_latents(vae, paths: list[str], size: int, device) -> torch.Tensor:
    latents = [((vae.encode(batch.to(device, DTYPE)).latent_dist.sample() - vae.config.shift_factor)
                * vae.config.scaling_factor).to(DTYPE)
               for batch in _image_batches(paths, size)]
    return torch.cat(latents)


@torch.no_grad()
def _check_zero_init(embedding, pooled, device) -> None:
    """Setting a value on a freshly initialized embedding must leave the conditioning unchanged.

    The check runs in the base model's dtype, so the MLP is cast to bfloat16 here; it is trained
    in float32 afterwards, starting from these bfloat16-rounded weights.
    """
    probe = torch.tensor([500.0], device=device)
    embedding.embedder.to(pooled.dtype)
    embedding.value = None
    plain = embedding(probe, pooled)
    embedding.value = torch.tensor([1.234], device=device)
    assert torch.equal(plain, embedding(probe, pooled)), "value embedding is not zero-initialized"
    embedding.value = None


def _flow_matching_loss(prediction, noise, x0, sigma, batch_size):
    from diffusers.training_utils import compute_loss_weighting_for_sd3

    weight = compute_loss_weighting_for_sd3("logit_normal", sigma)
    squared = weight.float() * (prediction.float() - (noise - x0).float()) ** 2
    return torch.mean(squared.reshape(batch_size, -1), 1).mean()


def train(paths: list[str], prompt: str, out_dir: str | Path, *, steps: int, regressor=None,
          value_embedding: bool = True, rank: int = TRAINING["rank"], batch_size: int = TRAINING["batch_size"],
          learning_rate: float = TRAINING["learning_rate"], size: int = TRAINING["size"],
          seed: int = TRAINING["seed"], device: str = "cuda", log_every: int = 50) -> list[float]:
    """Train on ``paths`` and save the result to ``out_dir``; returns the loss of every step.

    ``regressor`` is required when ``value_embedding`` is True.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    from diffusers.training_utils import compute_density_for_timestep_sampling
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    if value_embedding and regressor is None:
        raise ValueError("the target-value embedding needs the regressor's pseudo-labels")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pseudo_label(paths, regressor, size, device) if regressor is not None else np.zeros(len(paths), np.float32)
    mean, std = float(labels.mean()), float(labels.std()) or 1.0
    if regressor is not None:
        regressor.to("cpu")   # not needed again; frees the GPU for training
        torch.cuda.empty_cache()

    # Encode the prompt and every training image once, then keep only the transformer.
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE_MODEL, revision=BASE_REVISION, torch_dtype=DTYPE).to(device)
    with torch.no_grad():
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                                                         device=device, num_images_per_prompt=1,
                                                         do_classifier_free_guidance=False)
    latents = _encode_latents(pipe.vae, paths, size, device)
    standardized = torch.tensor((labels - mean) / std, device=device, dtype=torch.float32)
    transformer = pipe.transformer
    embedding = attach_value_embedding(transformer).to(device) if value_embedding else None
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                                                                subfolder="scheduler")
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    pipe.vae = None
    del pipe
    torch.cuda.empty_cache()
    if embedding is not None:
        _check_zero_init(embedding, pooled, device)

    # Trainable parameters: the LoRA (float32 master weights) and, for ScoreDial, the value MLP.
    transformer.requires_grad_(False)
    transformer = get_peft_model(transformer, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0,
                                                         target_modules=LORA_TARGETS))
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if embedding is not None:
        embedding.embedder.to(torch.float32).requires_grad_(True)
    transformer.enable_gradient_checkpointing()
    transformer.train()
    lora_params = [p for n, p in transformer.named_parameters() if p.requires_grad and "embedder" not in n]
    value_params = list(embedding.embedder.parameters()) if embedding is not None else []
    trainable = lora_params + value_params
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    n_train = scheduler.config.num_train_timesteps
    sigmas = scheduler.sigmas.to(device, torch.float32)
    timesteps = scheduler.timesteps.to(device)
    prompt_batch = prompt_embeds.expand(batch_size, -1, -1)
    pooled_batch = pooled.expand(batch_size, -1)
    losses, started = [], time.time()
    for step in range(1, steps + 1):
        index = torch.randint(0, latents.shape[0], (batch_size,), device=device)
        x0 = latents[index].float()
        flip = (torch.rand(batch_size, device=device) < 0.5).view(-1, 1, 1, 1)   # horizontal flip, p = 0.5
        x0 = torch.where(flip, torch.flip(x0, dims=[-1]), x0)
        noise = torch.randn_like(x0)
        u = compute_density_for_timestep_sampling("logit_normal", batch_size, logit_mean=0.0, logit_std=1.0)
        t = timesteps[(u * n_train).long().clamp(0, n_train - 1)]
        sigma = sigmas[[(timesteps == ti).nonzero().item() for ti in t]].view(-1, 1, 1, 1)
        noisy = (1.0 - sigma) * x0 + sigma * noise
        if embedding is not None:
            embedding.value = standardized[index]
        with torch.autocast("cuda", dtype=DTYPE):
            prediction = transformer(hidden_states=noisy.to(DTYPE), timestep=t, encoder_hidden_states=prompt_batch,
                                     pooled_projections=pooled_batch.to(DTYPE), return_dict=False)[0]
        loss = _flow_matching_loss(prediction, noise, x0, sigma, batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == steps):
            print(f"step {step}/{steps} loss {loss.item():.4f} ({(time.time() - started) / step:.2f}s/it)", flush=True)

    transformer.save_pretrained(str(out_dir))
    if embedding is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in embedding.embedder.state_dict().items()},
                  str(out_dir / "value_embedding.safetensors"))
        (out_dir / "value_embedding.json").write_text(json.dumps({"mean": mean, "std": std}))
    (out_dir / "training.json").write_text(json.dumps(dict(
        prompt=prompt, n_images=len(paths), steps=steps, rank=rank, batch_size=batch_size,
        learning_rate=learning_rate, size=size, seed=seed, value_embedding=value_embedding,
        base_model=BASE_MODEL, base_revision=BASE_REVISION), indent=1))
    return losses
