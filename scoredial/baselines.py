"""The baselines (Numeric prompt, MPGD-flow, UG-flow) and how their guidance strength is chosen.

* Numeric prompt: the requested value is written into the prompt; plain CFG sampling
  (``scoredial.sampler.sample`` with guidance off).
* MPGD-flow: MPGD-LDM's Algorithm 2 adapted to rectified flow. Each step takes one gradient step of
  (R(x) - c)^2 with a fixed coefficient on the predicted clean latent; no back-propagation through
  the denoiser and no refinement.
* UG-flow: forward Universal Guidance adapted to rectified flow. The gradient is taken through
  the denoiser with respect to the noisy latent, with Universal Stepwise Refinement (two passes
  per step) and no backward optimization steps.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .config import ADMISSIBLE_STRENGTHS, STRENGTH_LADDER


def guided_sample(method: str, target: float, seed: int, strength: float, *, transformer, vae, scheduler,
                  text, regressor, steps: int = 30, cfg_scale: float = 5.0, refinements: int = 2, size: int = 1024,
                  device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """One image with MPGD-flow (``method="mpgd"``) or UG-flow (``"ug"``).

    Returns ``(image in [-1, 1], R score of the image)``.
    """
    if method not in ("mpgd", "ug"):
        raise ValueError(method)
    (conditional, pooled), (unconditional, unpooled) = text
    scheduler.set_timesteps(steps, device=device)
    sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
    generator = torch.Generator(device).manual_seed(seed)
    z = torch.randn(1, transformer.config.in_channels, size // 8, size // 8,
                    generator=generator, device=device, dtype=dtype) * sigmas[0]
    target_t = torch.tensor([target], device=device)
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor

    def velocity(latent, timestep, track_gradient):
        context = torch.enable_grad() if track_gradient else torch.no_grad()
        with context, torch.autocast("cuda", dtype=dtype):
            t = timestep.expand(latent.shape[0])
            vc = transformer(hidden_states=latent, timestep=t, encoder_hidden_states=conditional,
                             pooled_projections=pooled, return_dict=False)[0]
            vu = transformer(hidden_states=latent, timestep=t, encoder_hidden_states=unconditional,
                             pooled_projections=unpooled, return_dict=False)[0]
        return vu + cfg_scale * (vc - vu)

    def decode(clean):
        return vae.decode(clean / scale + shift).sample

    def squared_error(clean):
        return ((regressor(decode(clean).float()) - target_t) ** 2).sum()

    def mpgd_step(z, timestep, sigma, sigma_next):
        with torch.no_grad():
            v = velocity(z, timestep, False)
            clean = (z - sigma * v).to(dtype)
            noise = (z + (1.0 - sigma) * v).to(dtype)
        clean = clean.detach().requires_grad_(True)
        gradient = torch.autograd.grad(squared_error(clean), clean)[0]
        with torch.no_grad():
            guided = (clean.detach() - strength * gradient.float()).to(dtype)
            return ((1.0 - sigma_next) * guided + sigma_next * noise).to(dtype)

    def ug_step(z, timestep, sigma, sigma_next):
        if float(1.0 - sigma) < 1e-4:
            # At sigma = 1 the guided clean estimate would divide by zero: plain Euler step.
            with torch.no_grad():
                v = velocity(z, timestep, False)
                clean = (z - sigma * v).to(dtype)
                noise = (z + (1.0 - sigma) * v).to(dtype)
                return ((1.0 - sigma_next) * clean + sigma_next * noise).to(dtype)
        z_grad = z.detach().requires_grad_(True)
        v = velocity(z_grad, timestep, True)
        gradient = torch.autograd.grad(squared_error(z_grad - sigma * v), z_grad)[0]
        with torch.no_grad():
            noise = z_grad + (1.0 - sigma) * v
            # Universal Guidance scales the gradient by the noise coefficient, which is sigma
            # under rectified flow.
            guided_noise = noise + strength * sigma * gradient
            guided_clean = (z_grad - sigma * guided_noise) / (1.0 - sigma)
            return ((1.0 - sigma_next) * guided_clean + sigma_next * guided_noise).to(dtype)

    def refine(z_next, sigma, sigma_next):
        # Stepwise refinement: back to the current noise level, keeping the signal coefficient.
        ratio = float((1.0 - sigma) / (1.0 - sigma_next))
        fresh_scale = math.sqrt(max(float(sigma) ** 2 - (ratio * float(sigma_next)) ** 2, 0.0))
        fresh = torch.randn(z_next.shape, generator=generator, device=device, dtype=dtype)
        return (ratio * z_next + fresh_scale * fresh).to(dtype)

    for index, timestep in enumerate(timesteps):
        sigma = sigmas[index].to(device, dtype)
        sigma_next = sigmas[index + 1].to(device, dtype)
        if method == "mpgd":
            z = mpgd_step(z, timestep, sigma, sigma_next).detach()
            continue
        for repetition in range(refinements):
            z_next = ug_step(z, timestep, sigma, sigma_next)
            z = refine(z_next, sigma, sigma_next) if repetition < refinements - 1 else z_next.detach()

    with torch.no_grad():
        image = decode(z).float().clamp(-1, 1)
        score = regressor(image.float())
    return image, float(score[0].cpu())


def select_strength(method: str, endpoint_errors: dict[float, list[float]]) -> float:
    """The admissible strength with the lowest mean |R(x) - c| on the validation endpoints;
    ties go to the smaller strength.

    ``endpoint_errors[strength]`` holds R(x) - c on the saved images for c in {0.1, 0.9} and
    every validation seed.
    """
    admissible = ADMISSIBLE_STRENGTHS[method]
    means = {s: float(np.mean(np.abs(endpoint_errors[s]))) for s in admissible}
    return min(admissible, key=lambda s: (means[s], s))


def stronger_rung(method: str, selected: float) -> float:
    """The ladder strength one rung above ``selected`` (used for MPGD-flow+)."""
    position = STRENGTH_LADDER.index(selected)
    if position + 1 == len(STRENGTH_LADDER):
        raise ValueError(f"{method}: the selected strength {selected:g} is the top of the ladder; "
                         "there is no stronger rung")
    return STRENGTH_LADDER[position + 1]
