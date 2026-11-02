"""ScoreDial's sampler: target-conditioned classifier-free guidance plus error-adaptive regressor guidance.

The sampler runs SD3.5's rectified-flow Euler schedule. On guided steps it decodes the predicted
clean latent, scores it with the regressor R, and moves the clean latent along the normalized
gradient of (R(x) - c)^2 by a step proportional to the remaining error |R(x) - c| and to a weight
that decreases with the noise level. Each guided step makes two passes; between them the state is
re-noised above the current level, keeping the noise already predicted and drawing only the rest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GuidanceConfig:
    steps: int = 30             # Euler steps
    cfg_scale: float = 5.0      # classifier-free guidance scale
    lambda_main: float = 0.4    # guidance coefficient while start <= step/steps <= end
    lambda_tail: float = 0.05   # guidance coefficient after end
    start: float = 0.2          # no regressor guidance before this fraction of the schedule
    end: float = 0.8
    passes: int = 2             # guidance passes per guided step
    w_min: float = 0.2          # noise weight w(sigma) = w_min + (1 - w_min) * sigma**q
    q: float = 1.0
    backstep: float = 2.0       # between passes, re-noise to sigma_next + backstep * (sigma - sigma_next)
    # Switches for ablations.
    normalize_gradient: bool = True   # False: step along the raw gradient
    error_factor: bool = True         # False: drop |R(x) - c| from the step size


def guidance_coefficient(progress: float, guidance: GuidanceConfig) -> float:
    """Guidance coefficient at a fraction ``progress`` of the schedule."""
    if guidance.start <= progress <= guidance.end:
        return guidance.lambda_main
    if progress > guidance.end:
        return guidance.lambda_tail
    return 0.0


@torch.no_grad()
def _velocity(transformer, latent, timestep, text, cfg_scale, dtype):
    (cond, cond_pooled), (uncond, uncond_pooled) = text
    t = timestep.expand(latent.shape[0])
    with torch.autocast("cuda", dtype=dtype):
        v_cond = transformer(hidden_states=latent, timestep=t, encoder_hidden_states=cond,
                             pooled_projections=cond_pooled, return_dict=False)[0]
        v_uncond = transformer(hidden_states=latent, timestep=t, encoder_hidden_states=uncond,
                               pooled_projections=uncond_pooled, return_dict=False)[0]
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def _corrected_clean(clean, grad, error, lam, sigma, guidance: GuidanceConfig):
    """Move the clean latent against the gradient of (R(x) - c)^2."""
    if not guidance.normalize_gradient:
        return clean - lam * grad.float()
    direction = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
    step = lam * error.abs() if guidance.error_factor else torch.full_like(error, lam)
    step = step * (guidance.w_min + (1.0 - guidance.w_min) * float(sigma) ** guidance.q)
    norm = clean.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return clean - step.view(-1, 1, 1, 1) * norm * direction


@torch.no_grad()
def _renoise(z_next, sigma, sigma_next, guidance: GuidanceConfig, generator, device, dtype):
    """Raise the noise level of ``z_next`` (at sigma_next) to sigma_back for the next pass.

    The predicted noise in ``z_next`` is kept and only the missing variance is drawn afresh.
    Returns the new state and its noise level.
    """
    # Capped below 1 so that kappa stays positive.
    back = min(0.999, float(sigma_next) + guidance.backstep * (float(sigma) - float(sigma_next)))
    kappa = (1.0 - back) / (1.0 - float(sigma_next))
    added = math.sqrt(max(back * back - (kappa * float(sigma_next)) ** 2, 0.0))
    fresh = torch.randn(z_next.shape, generator=generator, device=device)
    return (kappa * z_next.float() + added * fresh).to(dtype), back


def sample(target: float, seed: int, *, transformer, vae, scheduler, text, regressor,
           guidance: GuidanceConfig = GuidanceConfig(), n: int = 1, size: int = 1024,
           device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Generate ``n`` images whose regressor score should equal ``target``.

    ``text`` is ``((cond, cond_pooled), (uncond, uncond_pooled))``; ``regressor`` maps images in
    [-1, 1] to scores of shape ``(n,)``. The requested value must already be set on the
    transformer's value embedding (``scoredial.model.set_target``). With ``lambda_main =
    lambda_tail = 0`` there is no regressor guidance. Returns ``(images in [-1, 1], scores)``.
    """
    scheduler.set_timesteps(guidance.steps, device=device)
    sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
    generator = torch.Generator(device).manual_seed(seed)
    z = torch.randn(n, transformer.config.in_channels, size // 8, size // 8,
                    generator=generator, device=device, dtype=dtype) * sigmas[0]
    target_t = torch.full((n,), float(target), device=device)
    text = tuple((e.expand(n, -1, -1), p.expand(n, -1)) for e, p in text)
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor

    def decode(clean_latent):
        return vae.decode(clean_latent / scale + shift).sample

    for index, timestep in enumerate(timesteps):
        sigma = sigmas[index].to(device, dtype)
        sigma_next = sigmas[index + 1].to(device, dtype)
        lam = guidance_coefficient(index / len(timesteps), guidance)
        passes = max(1, guidance.passes) if lam > 0 else 1

        pass_sigma, pass_timestep = sigma, timestep
        for p in range(passes):
            v = _velocity(transformer, z, pass_timestep, text, guidance.cfg_scale, dtype)
            with torch.no_grad():
                clean = (z - pass_sigma * v).to(dtype)
                noise = (z + (1.0 - pass_sigma) * v).to(dtype)
            if lam == 0:
                z = ((1.0 - sigma_next) * clean + sigma_next * noise).to(dtype)
                continue

            clean = clean.detach().requires_grad_(True)
            error = regressor(decode(clean).float()) - target_t
            grad = torch.autograd.grad((error ** 2).sum(), clean)[0]
            guided = _corrected_clean(clean.detach(), grad, error, lam, pass_sigma, guidance).to(dtype)
            with torch.no_grad():
                z_next = ((1.0 - sigma_next) * guided + sigma_next * noise).to(dtype)
            if p < passes - 1:
                z, back = _renoise(z_next, sigma, sigma_next, guidance, generator, device, dtype)
                pass_sigma = torch.tensor(back, device=device, dtype=dtype)
                # The denoiser is conditioned on the schedule timestep nearest to sigma_back.
                pass_timestep = timesteps[int((sigmas[: len(timesteps)] - back).abs().argmin())]
            else:
                z = z_next

    with torch.no_grad():
        images = decode(z.to(dtype)).float().clamp(-1, 1)
        scores = regressor(images)
    return images, scores
