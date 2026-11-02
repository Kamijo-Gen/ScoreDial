"""Evaluation of generated grids: agreement with R, and attribute movement under an independent auditor.

* Agreement: R-MAE, the mean |R(x) - c| over a target x seed grid with R evaluated on the saved
  8-bit images, the population SD of the per-seed means, and Spearman rho between c and R(x).
* Movement: the auditor is a ridge head on frozen SigLIP features fitted to the same human ratings
  as R. For each seed it reads the images requested at the lowest and highest c; the endpoint
  displacement is A(c_max) - A(c_min).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

AUDITOR_BACKBONE = dict(model_id="google/siglip-base-patch16-224",
                        revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
AUDITOR_ALPHAS = (1.0, 10.0, 50.0, 100.0)
AUDITOR_CV_SEEDS = tuple(range(100, 105))


def load_png(path, device="cuda") -> torch.Tensor:
    """A saved image as R reads it for evaluation: (1, 3, H, W) in [-1, 1]."""
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def agreement(targets, scores, seeds) -> dict:
    """R-MAE, SD of the per-seed R-MAE, and Spearman rho for one grid."""
    targets, scores, seeds = map(np.asarray, (targets, scores, seeds))
    errors = np.abs(scores - targets)
    per_seed = np.asarray([errors[seeds == s].mean() for s in np.unique(seeds)])
    return dict(r_mae=float(errors.mean()), r_mae_seed_sd=float(per_seed.std()),
                spearman=float(spearmanr(targets, scores).statistic), n=int(len(errors)))


def load_siglip(device="cuda"):
    """The auditor's image encoder, as ``(processor, model)``."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(AUDITOR_BACKBONE["model_id"], revision=AUDITOR_BACKBONE["revision"])
    model = AutoModel.from_pretrained(AUDITOR_BACKBONE["model_id"],
                                      revision=AUDITOR_BACKBONE["revision"]).to(device).eval()
    return processor, model


@torch.no_grad()
def siglip_features(paths: list[str], device="cuda", batch_size: int = 16, siglip=None) -> np.ndarray:
    """SigLIP image embeddings of image files; pass ``siglip=load_siglip(device)`` to reuse a loaded encoder."""
    processor, model = siglip or load_siglip(device)
    embeddings = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
        output = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
        output = output if torch.is_tensor(output) else output.pooler_output
        embeddings.append(output.float().cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32)


def _r2(y, prediction):
    return 1.0 - np.square(y - prediction).sum() / (np.square(y - y.mean()).sum() + 1e-12)


def _fit_predict(X_train, y_train, X_test, alpha):
    scaler = StandardScaler().fit(X_train)
    return Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_test))


def _choose_alpha(X, y, seeds=AUDITOR_CV_SEEDS):
    """The ridge alpha with the highest mean out-of-fold R^2 over repeated 5-fold CV."""
    scores = defaultdict(list)
    for seed in seeds:
        for alpha in AUDITOR_ALPHAS:
            out_of_fold = np.empty_like(y)
            for train, test in KFold(5, shuffle=True, random_state=seed).split(X):
                out_of_fold[test] = _fit_predict(X[train], y[train], X[test], alpha)
            scores[alpha].append(_r2(y, out_of_fold))
    means = {a: float(np.mean(v)) for a, v in scores.items()}
    return max(means, key=means.get)


class Auditor:
    """A ridge head on SigLIP features, fitted to the human ratings."""

    def __init__(self, mean, scale, weight, bias, alpha):
        self.mean, self.scale = np.asarray(mean, np.float64), np.asarray(scale, np.float64)
        self.weight, self.bias, self.alpha = np.asarray(weight, np.float64), float(bias), float(alpha)

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> "Auditor":
        X, y = np.asarray(features, np.float64), np.asarray(labels, np.float64)
        alpha = _choose_alpha(X, y)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
        return cls(scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32),
                   model.coef_.astype(np.float32), model.intercept_, alpha)

    @staticmethod
    def nested_cv_r2(features: np.ndarray, labels: np.ndarray) -> dict:
        """Out-of-fold R^2 of the whole fitting procedure, alpha chosen inside each training fold."""
        X, y = np.asarray(features, np.float64), np.asarray(labels, np.float64)
        repeats = []
        for seed in AUDITOR_CV_SEEDS:
            prediction = np.empty_like(y)
            for train, test in KFold(5, shuffle=True, random_state=seed).split(X):
                alpha = _choose_alpha(X[train], y[train], seeds=[seed])
                prediction[test] = _fit_predict(X[train], y[train], X[test], alpha)
            repeats.append(_r2(y, prediction))
        return dict(r2_mean=float(np.mean(repeats)), r2_std=float(np.std(repeats)), r2_repeats=repeats)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, np.float64) - self.mean) / self.scale) @ self.weight + self.bias


def endpoint_displacement(auditor: Auditor, low_paths: list[str], high_paths: list[str], device="cuda",
                          siglip=None) -> dict:
    """A(high) - A(low) per seed, where ``low_paths[i]`` and ``high_paths[i]`` share seed i."""
    siglip = siglip or load_siglip(device)
    per_seed = []
    for low, high in zip(low_paths, high_paths):
        reading = auditor(siglip_features([low, high], device=device, siglip=siglip))
        per_seed.append(float(reading[1] - reading[0]))
    per_seed = np.asarray(per_seed)
    return dict(mean=float(per_seed.mean()), sd=float(per_seed.std()), per_seed=per_seed.tolist())
